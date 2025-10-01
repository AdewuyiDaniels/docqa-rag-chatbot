import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import VectorStoreManager

@pytest.fixture
def mock_embedding_function():
    """A mock for the embedding function dependency."""
    return MagicMock()

@pytest.fixture
def mock_chroma_client():
    """A mock for the ChromaDB client dependency."""
    client = MagicMock()
    collection = MagicMock()
    collection_meta = MagicMock()
    collection_meta.name = "documents"

    collection.get.return_value = {"metadatas": [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}, {"source": "doc1.pdf"}]}
    client.list_collections.return_value = [collection_meta]
    client.get_collection.return_value = collection
    return client

# Patch the Chroma class *where it is used* inside the src.embeddings module
@patch('src.embeddings.Chroma')
def test_vsm_initialization(mock_chroma_class, mock_chroma_client, mock_embedding_function):
    """Test that VectorStoreManager initializes correctly with injected dependencies."""
    manager = VectorStoreManager(client=mock_chroma_client, embedding_function=mock_embedding_function)

    assert manager.client == mock_chroma_client
    assert manager.embedding_function == mock_embedding_function
    mock_chroma_class.assert_called_once_with(
        client=mock_chroma_client,
        collection_name="documents",
        embedding_function=mock_embedding_function
    )

@patch('src.embeddings.Chroma')
def test_add_documents(mock_chroma_class, mock_chroma_client, mock_embedding_function):
    """Test the add_documents method."""
    mock_vector_store_instance = mock_chroma_class.return_value

    manager = VectorStoreManager(client=mock_chroma_client, embedding_function=mock_embedding_function)

    test_chunks = [{"text": "this is a test", "metadata": {"source": "test.txt"}}]
    manager.add_documents(test_chunks)

    mock_vector_store_instance.add_texts.assert_called_once_with(
        texts=['this is a test'],
        metadatas=[{'source': 'test.txt'}]
    )

@patch('src.embeddings.Chroma')
def test_get_retriever(mock_chroma_class, mock_chroma_client, mock_embedding_function):
    """Test the get_retriever method."""
    mock_vector_store_instance = mock_chroma_class.return_value

    manager = VectorStoreManager(client=mock_chroma_client, embedding_function=mock_embedding_function)
    manager.get_retriever()

    mock_vector_store_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 4})

@patch('src.embeddings.Chroma')
def test_clear_collection(mock_chroma_class, mock_chroma_client, mock_embedding_function):
    """Test the clear_collection method."""
    manager = VectorStoreManager(client=mock_chroma_client, embedding_function=mock_embedding_function)
    manager.clear_collection(collection_name="test_collection")
    mock_chroma_client.delete_collection.assert_called_once_with(name="test_collection")

@patch('src.embeddings.Chroma')
def test_get_processed_documents(mock_chroma_class, mock_chroma_client, mock_embedding_function):
    """Test retrieving processed document names."""
    manager = VectorStoreManager(client=mock_chroma_client, embedding_function=mock_embedding_function)
    docs = manager.get_processed_documents()

    assert docs == ["doc1.pdf", "doc2.pdf"]
    mock_chroma_client.get_collection.assert_called_once_with(name="documents")