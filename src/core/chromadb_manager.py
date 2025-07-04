import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from src.core.ollama_embedding import OllamaEmbedding


class ChromaDBManager:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = OllamaEmbedding()
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None  # We'll handle embeddings manually
        )
    
    def add_documents(self,
                      documents: List[str],
                      embeddings: List[List[float]],
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None) -> None:
        """
        Add documents to the ChromaDB collection with embeddings.
        
        Args:
            documents: List of text documents to add
            metadatas: Optional list of metadata dictionaries for each document
            ids: Optional list of unique IDs for each document
        """
        if not embeddings:
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Ensure metadatas is provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, 
              query_text: str, 
              n_results: int = 5,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the collection for similar documents.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary containing query results
        """
        # Generate embedding for query
        query_embedding = self.embedding_function.embed_query(query_text)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def add_single_document(self, 
                           document: str, 
                           metadata: Optional[Dict[str, Any]] = None,
                           doc_id: Optional[str] = None) -> None:
        """
        Add a single document to the collection.
        
        Args:
            document: Text document to add
            metadata: Optional metadata dictionary
            doc_id: Optional unique ID for the document
        """
        self.add_documents(
            documents=[document],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None
        )
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the collection by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents in the collection
        """
        return self.collection.count()
    
    def reset_collection(self) -> None:
        """
        Delete all documents from the collection.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None
        )

