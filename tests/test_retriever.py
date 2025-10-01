import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import RAGEngine

@pytest.fixture
def mock_retriever():
    """A mock for the vector store retriever."""
    return MagicMock()

@pytest.fixture
def mock_llm():
    """A mock for the language model."""
    return MagicMock()

# Patch the RetrievalQA chain *where it is used* inside the src.retriever module
@patch('src.retriever.RetrievalQA')
def test_rag_engine_initialization(mock_retrieval_qa_class, mock_retriever, mock_llm):
    """Test that RAGEngine initializes correctly and creates a QA chain."""
    engine = RAGEngine(retriever=mock_retriever, llm=mock_llm)

    assert engine.retriever == mock_retriever
    assert engine.llm == mock_llm
    # Verify that the from_chain_type class method was called on our mock
    mock_retrieval_qa_class.from_chain_type.assert_called_once()

@patch('src.retriever.RetrievalQA')
def test_rag_engine_query(mock_retrieval_qa_class, mock_retriever, mock_llm):
    """Test a standard query through the RAGEngine."""
    # Arrange: Configure the mock instance that will be returned by from_chain_type
    mock_qa_chain_instance = mock_retrieval_qa_class.from_chain_type.return_value
    mock_qa_chain_instance.invoke.return_value = {
        "result": "The sky is indeed blue.",
        "source_documents": [
            MagicMock(
                page_content="The sky is blue.",
                metadata={"source": "sky.pdf", "chunk_id": 1}
            )
        ]
    }

    engine = RAGEngine(retriever=mock_retriever, llm=mock_llm)

    # Act: Call the query method
    question = "What color is the sky?"
    response = engine.query(question)

    # Assert: Verify the chain was called and the response is correct
    mock_qa_chain_instance.invoke.assert_called_with({"query": question})
    assert response["answer"] == "The sky is indeed blue."
    assert len(response["sources"]) == 1
    assert response["sources"][0]["source"] == "sky.pdf"

@patch('src.retriever.RetrievalQA')
def test_rag_engine_empty_query(mock_retrieval_qa_class, mock_retriever, mock_llm):
    """Test that an empty query is handled gracefully."""
    mock_qa_chain_instance = mock_retrieval_qa_class.from_chain_type.return_value

    engine = RAGEngine(retriever=mock_retriever, llm=mock_llm)

    response = engine.query("")

    assert "Please ask a question" in response["answer"]
    assert len(response["sources"]) == 0
    mock_qa_chain_instance.invoke.assert_not_called()