"""
Handles document processing: text extraction from PDF and DOCX files,
and text chunking.
"""

import os
from typing import List, Dict, Union
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The extracted text, with excess whitespace removed.
    """
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return " ".join(text.split())
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        file_path: The path to the DOCX file.

    Returns:
        The extracted text, with excess whitespace removed.
    """
    try:
        document = docx.Document(file_path)
        text = "\n".join([para.text for para in document.paragraphs])
        return " ".join(text.split())
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
        return ""


def chunk_text(text: str, source_filename: str) -> List[Dict]:
    """
    Splits text into chunks with metadata.

    Args:
        text: The input text to be chunked.
        source_filename: The name of the source document.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        and includes the text, source, and chunk ID.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    chunked_docs = []
    for i, chunk in enumerate(chunks):
        chunked_docs.append({
            "text": chunk,
            "metadata": {
                "source": source_filename,
                "chunk_id": i
            }
        })
    return chunked_docs


def process_document(uploaded_file: UploadedFile, upload_dir: str) -> List[Dict]:
    """
    Processes an uploaded document (PDF or DOCX) by extracting and chunking its text.

    Args:
        uploaded_file: The file uploaded via Streamlit.
        upload_dir: The directory to temporarily store the uploaded file.

    Returns:
        A list of chunked documents with metadata, or an empty list if processing fails.
    """
    filename = uploaded_file.name
    file_path = os.path.join(upload_dir, filename)
    file_extension = os.path.splitext(filename)[1].lower()

    # Save the uploaded file temporarily
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = ""
    try:
        if file_extension == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_extension == ".docx":
            text = extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

        if not text.strip():
            print(f"Warning: No text extracted from {filename}.")
            return []

        chunks = chunk_text(text, filename)
        return chunks

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        return []
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)