import os
import sys
from unittest.mock import MagicMock

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_processor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    chunk_text,
    process_document
)
from src.config import UPLOAD_DIR

def test_extract_text_from_pdf(test_pdf_path):
    """Verify text extraction from a PDF file."""
    text = extract_text_from_pdf(test_pdf_path)
    assert "The sky is blue" in text
    assert "test document for PDF extraction" in text

def test_extract_text_from_docx(test_docx_path):
    """Verify text extraction from a DOCX file."""
    text = extract_text_from_docx(test_docx_path)
    assert "The grass is green" in text
    assert "test document for DOCX extraction" in text

def test_extract_text_from_empty_docx(empty_docx_path):
    """Ensure empty documents return an empty string without errors."""
    text = extract_text_from_docx(empty_docx_path)
    assert text == ""

def test_chunk_text():
    """Verify text chunking logic."""
    sample_text = "a" * 2000  # Text longer than chunk size
    chunks = chunk_text(sample_text, "sample.txt")
    assert len(chunks) > 1
    assert chunks[0]['metadata']['source'] == "sample.txt"
    assert chunks[0]['metadata']['chunk_id'] == 0
    assert chunks[1]['metadata']['chunk_id'] == 1

def test_chunk_empty_text():
    """Ensure chunking empty text returns an empty list."""
    chunks = chunk_text("", "empty.txt")
    assert len(chunks) == 0

def test_process_document_pdf(test_pdf_path):
    """Test the end-to-end processing for a PDF file."""
    filename = os.path.basename(test_pdf_path)
    with open(test_pdf_path, 'rb') as f:
        file_bytes = f.read()

    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = filename
    mock_uploaded_file.getbuffer.return_value = file_bytes

    chunks = process_document(mock_uploaded_file, UPLOAD_DIR)
    assert len(chunks) > 0
    assert "The sky is blue" in chunks[0]['text']

def test_process_document_unsupported_file():
    """Test handling of an unsupported file type."""
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "image.jpg"
    mock_uploaded_file.getbuffer.return_value = b"someimagedata"

    chunks = process_document(mock_uploaded_file, UPLOAD_DIR)
    assert len(chunks) == 0