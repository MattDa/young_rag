import os
import psycopg2
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import tiktoken
from typing import List, Dict, Tuple

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "1qaz",
}

PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "documents"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_MODEL = "gpt-5"


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def split_large_chunk(text: str) -> List[str]:
    if token_count(text) <= 1000:
        return [text]

    parts = [text]
    while parts:
        current = parts.pop(0)
        if token_count(current) <= 1000:
            yield current
            continue

        midpoint = len(current) // 2
        parts.insert(0, current[midpoint:])
        parts.insert(0, current[:midpoint])


def apply_overlap(chunks: List[str], overlap_ratio: float = 0.2) -> List[str]:
    tokenizer = get_tokenizer()
    if not chunks:
        return []

    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prev_tokens = tokenizer.encode(previous)
        overlap_size = max(1, int(len(prev_tokens) * overlap_ratio))
        overlap_tokens = prev_tokens[-overlap_size:]
        overlapped.append(tokenizer.decode(overlap_tokens + tokenizer.encode(current)))
    return overlapped


def chunk_text(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buffer: List[str] = []

    for idx, para in enumerate(paragraphs):
        buffer.append(para)
        combined = "\n\n".join(buffer)
        if token_count(combined) >= 300:
            chunks.append(combined)
            buffer = []

    if buffer:
        chunks.append("\n\n".join(buffer))

    normalized_chunks: List[str] = []
    for chunk in chunks:
        if token_count(chunk) > 1500:
            normalized_chunks.extend(list(split_large_chunk(chunk)))
        else:
            normalized_chunks.append(chunk)

    return apply_overlap(normalized_chunks, 0.2)


def fetch_rows() -> List[Tuple[str, str]]:
    connection = psycopg2.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT path, text FROM documents")
            return cursor.fetchall()
    finally:
        connection.close()


@st.cache_resource(show_spinner=False)
def get_vector_store():
    embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )


def build_documents(rows: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    documents = []
    for path, text in rows:
        for index, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{path}-{index}",
                    "text": chunk,
                    "metadata": {"path": path, "chunk_index": index},
                }
            )
    return documents


def sync_vector_store() -> bool:
    rows = fetch_rows()
    vector_store = get_vector_store()
    collection = vector_store._collection

    existing_count = collection.count()
    existing = collection.get(limit=existing_count, include=["ids"]) if existing_count else {"ids": []}
    existing_ids = set(existing.get("ids", []))

    documents = build_documents(rows)
    new_docs = [doc for doc in documents if doc["id"] not in existing_ids]

    if not new_docs:
        return False

    vector_store.add_texts(
        texts=[doc["text"] for doc in new_docs],
        metadatas=[doc["metadata"] for doc in new_docs],
        ids=[doc["id"] for doc in new_docs],
    )
    return True


def retrieve_context(query: str, top_k: int = 3):
    retriever = get_vector_store().as_retriever(search_kwargs={"k": top_k})
    return retriever.get_relevant_documents(query)


def stream_answer(question: str, docs):
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        streaming=True,
        temperature=1,
        # openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    context_lines = []
    for idx, doc in enumerate(docs, start=1):
        context_lines.append(f"[{idx}] {doc.page_content}\nSource: {doc.metadata.get('path', 'unknown')}")
    context = "\n\n".join(context_lines) if context_lines else "No relevant context found."

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. Use the provided context to answer the question. "
            "Include citations for the top 3 retrieved chunks in the format: 'Sources: - path (chunk id)'."
        )
    )
    human_message = HumanMessage(
        content=(
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Provide a concise answer with citations."
        )
    )

    response_placeholder = st.empty()
    streamed_text = ""
    with st.spinner("Generating response..."):
        for chunk in llm.stream([system_message, human_message]):
            if chunk.content:
                streamed_text += chunk.content
                response_placeholder.markdown(streamed_text)

    citation_lines = [
        f"- {doc.metadata.get('path', 'unknown')} (chunk {doc.metadata.get('chunk_index', 'n/a')})"
        for doc in docs
    ]
    if citation_lines:
        streamed_text += "\n\nSources:\n" + "\n".join(citation_lines)
        response_placeholder.markdown(streamed_text)

    return streamed_text


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="RAG Chat", page_icon="💬")
    st.markdown(
        """
        <style>
            .stApp {background-color: #0E1117; color: #E0E0E0;}
            .stTextInput>div>div>input {background-color: #1E1E1E; color: #E0E0E0;}
            .stButton>button {background-color: #2A2A2A; color: #E0E0E0; border: 1px solid #444;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("LangChain RAG Chat")
    init_session_state()

    if st.button("Check for new data"):
        with st.spinner("Updating vector store..."):
            updated = sync_vector_store()
        if updated:
            st.success("Successfully updated")
        else:
            st.info("No new data found")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            docs = retrieve_context(question)
            answer = stream_answer(question, docs)
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()