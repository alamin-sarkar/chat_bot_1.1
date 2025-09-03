# pinecone and groq

import os
import dotenv
from langchain_groq import ChatGroq
import hashlib, streamlit as st
from pinecone import Pinecone,ServerlessSpec
from sentence_transformers import SentenceTransformer

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
dotenv.load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chat_bot_1"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index_name = 'chat-bot-1-history-index'
index = pc.Index(index_name)
print('pinecone activated')
llm = ChatGroq(model='openai/gpt-oss-20b', groq_api_key=groq_api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_content_id(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def embed_texts(texts):
    return embedder.encode(texts).tolist()

def get_content_id(content: str):
    """Generate a unique hash for the content to avoid duplicates"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def save_messages(session_id, messages):
    """
    Save multiple messages (user + assistant) to Pinecone in batch
    messages = [(role, content), ...]
    """
    new_msgs = []
    if "saved_ids" not in st.session_state:
        st.session_state.saved_ids = set()

    # Filter out duplicates
    for role, content in messages:
        cid = get_content_id(content)
        if cid not in st.session_state.saved_ids:
            new_msgs.append((role, content, cid))

    if not new_msgs:
        return  # nothing new to save

    # Batch embed
    contents = [c for _, c, _ in new_msgs]
    vectors = embedder.encode(contents).tolist()  # MiniLM embeddings

    # Prepare upserts
    upserts = []
    for (role, content, cid), vec in zip(new_msgs, vectors):
        upserts.append({
            "id": f"{session_id}-{role}-{cid}",
            "values": vec,
            "metadata": {"session_id": session_id, "role": role, "content": content}
        })
        st.session_state.saved_ids.add(cid)

    # Upsert to Pinecone in batch
    index.upsert(upserts)
    print(f"✅ Saved {len(upserts)} new messages for session {session_id}")


def save_message_if_new(session_id, role, content, threshold=0.9):
    # 1) Embed new content
    vector = embedder.encode([content])[0].tolist()

    # 2) Query Pinecone for similar messages in this session
    result = index.query(
        vector=vector,
        filter={"session_id": {"$eq": session_id}},
        top_k=1,
        include_metadata=True
    )

    # 3) Check similarity
    if result['matches']:
        sim = result['matches'][0]['score']  # cosine similarity
        if sim >= threshold:
            print("⚠️ Contextually similar message exists, skipping save")
            return

    # 4) Save new message
    content_id = hashlib.md5(content.encode("utf-8")).hexdigest()
    index.upsert([{
        "id": f"{session_id}-{role}-{content_id}",
        "values": vector,
        "metadata": {"session_id": session_id, "role": role, "content": content}
    }])
    print("✅ Saved new message")


# --- Retrieve messages ---
def get_relevant_history(session_id, query, top_k=3):
    vector = embed_texts([query])[0]
    result = index.query(
        vector=vector,
        filter={"session_id": {"$eq": session_id}},
        top_k=top_k,
        include_metadata=True
    )
    return [m["metadata"]["content"] for m in result["matches"]]



# --- Streamlit UI ---
st.title("Chat bot with Pinecone Memory")

session_id = "chat_1"
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("Write your question here...")
with col2:
    selected_language = st.selectbox("Language", ["English", "Bangla"])

if user_input:
    # 1) Retrieve relevant history
    past_context = get_relevant_history(session_id, user_input)

    # 2) Make prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You're a helpful assistant. Answer in {selected_language}. "
                   f"Here is the past conversation context:\n{past_context}"),
        ("human", "{query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"query": user_input})

    # 3) Show answer
    st.write(response.content)

    save_message_if_new(session_id=session_id, role="user", content=user_input)
    save_message_if_new(session_id=session_id, role="assistant", content=response.content)
