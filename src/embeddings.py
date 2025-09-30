"""
Manages the vector store, including embedding generation, document addition,
and retriever creation.
"""

from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStoreRetriever
import chromadb

from src.config import VECTOR_STORE_DIR, TOP_K_RESULTS


class VectorStoreManager:
    """
    Handles interactions with the ChromaDB vector store.
    """

    def __init__(self, persist_directory: str = VECTOR_STORE_DIR):
        """
        Initializes the VectorStoreManager.

        Args:
            persist_directory: The directory where the vector store will be persisted.
        """
        self.persist_directory = persist_directory
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = Chroma(
            client=self.client,
            collection_name="documents",
            embedding_function=self.embedding_function,
        )

    def add_documents(self, chunks: List[Dict], collection_name: str = "documents"):
        """
        Adds document chunks to the specified collection in the vector store.

        Args:
            chunks: A list of chunk dictionaries, each with 'text' and 'metadata'.
            collection_name: The name of the collection to add documents to.
        """
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        self.vector_store.persist()
        print(f"Added {len(chunks)} chunks to collection '{collection_name}'.")

    def get_retriever(self, k: int = TOP_K_RESULTS) -> VectorStoreRetriever:
        """
        Gets a retriever for the vector store.

        Args:
            k: The number of top results to retrieve.

        Returns:
            A retriever instance.
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def clear_collection(self, collection_name: str = "documents"):
        """
        Clears all documents from a specified collection.

        Args:
            collection_name: The name of the collection to clear.
        """
        try:
            self.client.delete_collection(name=collection_name)
            self.vector_store = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding_function,
            )
            print(f"Cleared all documents from collection '{collection_name}'.")
        except ValueError:
            print(f"Collection '{collection_name}' does not exist, nothing to clear.")
        except Exception as e:
            print(f"An error occurred while clearing collection '{collection_name}': {e}")

    def get_processed_documents(self, collection_name: str = "documents") -> List[str]:
        """
        Retrieves a list of unique source document names from the vector store.

        Returns:
            A list of unique source document filenames.
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            if not collection:
                return []

            metadatas = collection.get(include=["metadatas"])["metadatas"]
            if not metadatas:
                return []

            sources = {meta.get("source") for meta in metadatas if meta.get("source")}
            return sorted(list(sources))
        except ValueError:
             return []
        except Exception as e:
            print(f"Error retrieving processed documents: {e}")
            return []