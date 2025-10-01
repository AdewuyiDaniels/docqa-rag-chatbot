"""
Main Streamlit application for the Document Q&A Chatbot.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

# Import dependencies for the Composition Root
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb

from src.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, VECTOR_STORE_DIR
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

def load_css(file_path):
    """Loads a CSS file and injects it into the Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")

# Apply custom CSS
load_css("src/styles/style.css")

st.title("📄 Document Q&A Chatbot with RAG")

# --- DEPENDENCY INJECTION & COMPOSITION ROOT ---

@st.cache_resource
def setup_application():
    """
    Initializes and wires up all application dependencies.
    This function acts as the Composition Root for the application and is cached
    by Streamlit to avoid re-creating objects on every interaction.
    """
    # 1. Initialize core components
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 2. Compose the VectorStoreManager with its dependencies
    vector_store_manager = VectorStoreManager(
        client=chroma_client,
        embedding_function=embedding_function
    )

    # 3. Compose the RAGEngine with its dependencies
    retriever = vector_store_manager.get_retriever()
    rag_engine = RAGEngine(retriever=retriever, llm=llm)

    return vector_store_manager, rag_engine

# --- STATE MANAGEMENT ---

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload your documents and I'll be ready to answer your questions."}
    ]
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

# Setup the application and get the main components
vector_store_manager, rag_engine = setup_application()

# Update processed documents list on every run to reflect changes
st.session_state.processed_docs = vector_store_manager.get_processed_documents()


# --- HELPER FUNCTIONS ---

def display_chat_history():
    """Displays the chat history without the source expanders."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_sources(sources_list):
    """Renders the source documents in the dedicated sources column."""
    with sources_placeholder.container():
        if not sources_list:
            st.info("Source documents will appear here when you ask a question.")
        else:
            st.subheader("Source Documents")
            for source in sources_list:
                with st.expander(f"Source: {source['source']} (Chunk {source['chunk_id']})"):
                    st.markdown(source['content'])

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
    # Force a rerun to update the processed docs list in the UI
    st.rerun()

def clear_all_data():
    """Clears the vector store, chat history, and processed documents list."""
    with st.spinner("Clearing all data..."):
        vector_store_manager.clear_collection()
        st.session_state.messages = []
        st.session_state.processed_docs = []
        # Invalidate cached resources to force re-initialization
        st.cache_resource.clear()
    st.success("All documents and chat history have been cleared.")
    st.rerun()


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

# --- MAIN UI LAYOUT ---
chat_col, sources_col = st.columns([0.65, 0.35])

with sources_col:
    st.header("Sources")
    sources_placeholder = st.container()
    # Display sources from the latest query
    display_sources(st.session_state.latest_sources)

with chat_col:
    # --- MAIN CHAT UI ---
    display_chat_history()

    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the previous messages for context, but not the current prompt
                chat_history = st.session_state.messages[:-1]
                response = rag_engine.query(prompt, chat_history)
                answer = response["answer"]
                st.session_state.latest_sources = response["sources"]

                st.markdown(answer)

        # Add assistant's response to history (without sources)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # Rerun to update the sources column
        st.rerun()