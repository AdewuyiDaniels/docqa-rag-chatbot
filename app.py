"""
Main Streamlit application for the Document Q&A Chatbot.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.config import UPLOAD_DIR, ALLOWED_EXTENSIONS
from src.document_processor import process_document
from src.embeddings import VectorStoreManager
from src.retriever import RAGEngine

# --- INITIALIZATION ---

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY environment variable not set. Please create a .env file.")
    st.stop()

st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("📄 Document Q&A Chatbot with RAG")

# --- STATE MANAGEMENT & CACHING ---

@st.cache_resource
def get_vector_store_manager():
    """Initializes and returns the VectorStoreManager."""
    return VectorStoreManager()

@st.cache_resource
def get_rag_engine(_vector_store_manager):
    """Initializes and returns the RAGEngine."""
    return RAGEngine(_vector_store_manager)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

# Get cached resources
vector_store_manager = get_vector_store_manager()
rag_engine = get_rag_engine(vector_store_manager)

# Update processed documents list on start
st.session_state.processed_docs = vector_store_manager.get_processed_documents()


# --- HELPER FUNCTIONS ---

def display_chat_history():
    """Displays the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.info(f"**Source:** {source['source']} (Chunk {source['chunk_id']})")
                        st.code(source['content'])

def handle_document_processing(uploaded_files):
    """Processes uploaded documents and adds them to the vector store."""
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return

    with st.spinner("Processing documents... This may take a moment."):
        total_chunks = 0
        processing_times = {}

        for uploaded_file in uploaded_files:
            start_time = time.time()

            # Process each document
            chunks = process_document(uploaded_file, UPLOAD_DIR)

            if chunks:
                vector_store_manager.add_documents(chunks)
                total_chunks += len(chunks)
                if uploaded_file.name not in st.session_state.processed_docs:
                    st.session_state.processed_docs.append(uploaded_file.name)

            end_time = time.time()
            processing_times[uploaded_file.name] = end_time - start_time

    st.success(f"Processed {len(uploaded_files)} documents, creating {total_chunks} chunks.")
    for name, duration in processing_times.items():
        st.info(f"'{name}' processed in {duration:.2f} seconds.")

def clear_all_data():
    """Clears the vector store, chat history, and processed documents list."""
    with st.spinner("Clearing all data..."):
        vector_store_manager.clear_collection()
        st.session_state.messages = []
        st.session_state.processed_docs = []
        # Invalidate cached resources to force re-initialization if needed
        st.cache_resource.clear()
    st.success("All documents and chat history have been cleared.")


# --- SIDEBAR UI ---

with st.sidebar:
    st.header("📚 Document Management")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=[ext.strip('.') for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        help="Supports PDF and DOCX files."
    )

    if st.button("Process Documents", use_container_width=True):
        handle_document_processing(uploaded_files)

    st.header("📖 Processed Documents")
    if not st.session_state.processed_docs:
        st.info("No documents have been processed yet.")
    else:
        for doc_name in st.session_state.processed_docs:
            st.markdown(f"- `{doc_name}`")

    st.header("⚠️ Danger Zone")
    if st.button("Clear All Data", use_container_width=True, type="primary"):
        clear_all_data()

# --- MAIN CHAT UI ---

display_chat_history()

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_engine.query(prompt)
            answer = response["answer"]
            sources = response["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.info(f"**Source:** {source['source']} (Chunk {source['chunk_id']})")
                        st.code(source['content'])

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })