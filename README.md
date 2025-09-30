# Document Q&A Chatbot with RAG

This project is a functional, lightweight Q&A chatbot that answers user questions based on the content of uploaded documents. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers with source citations directly from the provided texts.

The application is built with Python and Streamlit, using LangChain for RAG orchestration and ChromaDB for efficient, local vector storage.

## ✨ Features

- **Multi-Format Document Support**: Upload and process both PDF (`.pdf`) and Microsoft Word (`.docx`) files.
- **RAG-Powered Q&A**: Asks questions in natural language and get answers synthesized from the document content.
- **Source Citations**: Every answer is backed by citations from the original documents, allowing you to verify the information's source.
- **Session-Based Chat**: Chat history is maintained throughout your session.
- **Easy Data Management**: Upload new documents, view a list of processed files, and clear all data with the click of a button.
- **Local First**: Uses a file-based vector store (ChromaDB), requiring no external database setup.
- **Simple UI**: A clean and intuitive user interface built with Streamlit.

## 🏗️ Architecture Overview

The application follows a modular structure, separating concerns into distinct components:

1.  **Streamlit UI (`app.py`)**: The user-facing application that handles file uploads, displays the chat interface, and orchestrates the backend services.
2.  **Document Processor (`src/document_processor.py`)**: Responsible for extracting raw text from uploaded PDF and DOCX files and splitting it into smaller, manageable chunks for embedding.
3.  **Vector Store Manager (`src/embeddings.py`)**: Manages the entire lifecycle of the vector database. It uses OpenAI's `text-embedding-3-small` model to convert text chunks into vector embeddings and stores them in a persistent ChromaDB collection on disk.
4.  **RAG Engine (`src/retriever.py`)**: This is the core of the chatbot. It takes a user's query, retrieves the most relevant document chunks from the vector store (similarity search), and feeds this context along with the original question to an LLM (GPT-3.5-Turbo) to generate a final, context-aware answer.

The flow is as follows:
`Upload -> Process & Chunk -> Embed & Store -> Query -> Retrieve -> Generate Answer`

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.10+
- An OpenAI API Key

### 1. Clone the Repository

```bash
git clone <repository-url>
cd document-qa-chatbot
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

The application requires an OpenAI API key to generate embeddings and answers.

1.  Make a copy of the example `.env.example` file and name it `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Open the new `.env` file in a text editor and add your OpenAI API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```
    **Note**: Never commit your `.env` file to version control. The included `.gitignore` file is already configured to prevent this.

### 5. Run the Application

Once the setup is complete, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 💻 How to Use

1.  **Upload Documents**: Use the file uploader in the sidebar to select one or more PDF or DOCX files.
2.  **Process Documents**: Click the "Process Documents" button. The app will extract text, create embeddings, and store them in the local vector database. You will see a list of processed documents in the sidebar.
3.  **Ask Questions**: Type your question into the chat input at the bottom of the main screen and press Enter.
4.  **View Answers and Sources**: The chatbot will provide an answer. You can expand the "View Sources" section below the answer to see the exact text chunks from the documents that were used to generate the response.
5.  **Clear Data**: If you wish to start over, click the "Clear All Data" button in the sidebar. This will remove all processed documents and clear the chat history.

## 🔧 Troubleshooting

-   **`OPENAI_API_KEY not set` Error**: Ensure you have created a `.env` file (not `.env.example`) in the root directory and that it contains your valid OpenAI API key.
-   **Slow Processing**: Processing large documents can take time, as each chunk needs to be sent to the OpenAI API for embedding. Please be patient.
-   **No Answer or "I don't know"**: If the chatbot cannot find relevant information in the uploaded documents, it is designed to state that it doesn't know the answer. This is a feature to prevent it from making up incorrect information (hallucinating). Try rephrasing your question or uploading a document that contains the relevant context.
-   **Corrupted Files**: If a PDF or DOCX file is corrupted or cannot be read, the document processor will print an error to the console and skip that file.
-   **Dependencies Issues**: If you encounter errors after installation, try recreating the virtual environment and reinstalling the dependencies from `requirements.txt`.