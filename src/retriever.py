"""
The RAG engine that orchestrates the retrieval and generation process with conversational memory.
"""

from typing import Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage


class RAGEngine:
    """
    The main engine for handling conversational RAG queries.
    It is initialized with a retriever and a language model.
    """

    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        """
        Initializes the RAGEngine.

        Args:
            retriever: An initialized vector store retriever.
            llm: An initialized language model.
        """
        self.retriever = retriever
        self.llm = llm
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            # We can add a custom prompt for the document combination part if needed
        )

    def query(self, question: str, chat_history: List[Dict]) -> Dict:
        """
        Queries the conversational RAG pipeline.

        Args:
            question: The user's question.
            chat_history: The history of the conversation.

        Returns:
            A dictionary containing the answer and a list of source documents.
        """
        if not question:
            return {"answer": "Please ask a question.", "sources": []}

        # Format chat history from Streamlit's format to LangChain's format
        formatted_history = []
        for message in chat_history:
            if message["role"] == "user":
                formatted_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))

        try:
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": formatted_history
            })

            # The key for the answer in this chain is 'answer'
            answer = result.get("answer", "No answer found.")
            source_docs = result.get("source_documents", [])

            sources = []
            for doc in source_docs:
                metadata = doc.metadata
                sources.append({
                    "source": metadata.get("source", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                    "content": doc.page_content,
                })

            return {"answer": answer, "sources": sources}
        except Exception as e:
            print(f"Error during query processing: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
            }