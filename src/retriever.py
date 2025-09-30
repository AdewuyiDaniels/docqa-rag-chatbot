"""
The RAG engine that orchestrates the retrieval and generation process.
"""

from typing import Dict, List
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStoreRetriever

from src.embeddings import VectorStoreManager


class RAGEngine:
    """
    The main engine for handling RAG-based queries.
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initializes the RAGEngine.

        Args:
            vector_store_manager: An instance of VectorStoreManager.
        """
        self.retriever = vector_store_manager.get_retriever()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> RetrievalQA:
        """
        Creates the full question-answering chain.
        """
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide the answer and then list the sources used with their chunk ID.

        Context:
        {context}

        Question: {question}

        Helpful Answer:
        """
        QA_PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT},
        )

    def query(self, question: str) -> Dict:
        """
        Queries the RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            A dictionary containing the answer and a list of source documents.
        """
        if not question:
            return {"answer": "Please ask a question.", "sources": []}

        try:
            result = self.qa_chain.invoke({"query": question})

            answer = result.get("result", "No answer found.")
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