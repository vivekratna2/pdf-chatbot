import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

from src.core.chromadb_manager import ChromaDBManager
from src.core.ollama_embedding import OllamaEmbedding
from src.core.ollama_chat import OllamaChat
from src.utils.file_chunker import PDFChunker


class OllamaRAG:
    def __init__(
        self, 
        embedding_model: str = os.getenv("MODEL_NAME", "mistral"),
        chat_model: str = os.getenv("MODEL_NAME", "mistral"),
        base_url: str = "http://ollama:11434",
        top_k: int = 5,
        collection_name: str = "documents",
        chroma_db_path: str = "./chroma_db"
    ):
        # Keep OllamaEmbedding for generating embeddings
        self.embedding_client = OllamaEmbedding(embedding_model, base_url)
        self.chat_client = OllamaChat(chat_model, base_url)
        self.top_k = top_k
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = ChromaDBManager(collection_name = "resume_collection")
        
        # Create or get collection
        # self.collection = self.chroma_client.get_or_create_collection(
        #     name=collection_name,
        #     metadata={"hnsw:space": "cosine"}
        # )
    
    def add_documents(self, file_path: str, metadata: Optional[List[Dict]] = None) -> None:
        """Add documents to the knowledge base"""
        pdfchunker = PDFChunker(chunk_size=500)
        chunks = pdfchunker.process_pdf("/app/raw/Resume.pdf")
        
        # Generate embeddings using Ollama
        embeddings = self.embedding_client.embed_documents(chunks)
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = metadata if metadata else [{"source": f"doc_{i}"} for i in range(len(chunks))]
        
        # Store in ChromaDB
        # self.collection.add(
        #     documents=chunks,
        #     embeddings=embeddings,
        #     ids=ids,
        #     metadatas=metadatas
        # )
        self.chroma_client.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Retrieve most relevant documents for a query"""
        k = top_k if top_k is not None else self.top_k
        
        results = self.chroma_client.query(
            query_text=query,
        )
        
        # Format results
        relevant_docs = []
        if results['documents'] and results['distances']:
            for doc, distance in zip(results['documents'][0], results['distances'][0]):
                # Convert distance to similarity (ChromaDB returns distances, not similarities)
                similarity = 1 - distance
                relevant_docs.append((doc, similarity))
        
        return relevant_docs
    
    def generate_answer(
        self,
        user_question: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate answer using RAG approach"""
        
        # Retrieve relevant documents from ChromaDB
        relevant_docs = self.retrieve_relevant_documents(user_question)
        
        if not relevant_docs:
            return {
                "answer": "I don't have relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Extract context texts
        context_texts = [doc[0] for doc in relevant_docs]
        
        # Generate answer using Ollama
        answer = self.chat_client.generate_answer(
            user_question=user_question,
            context=context_texts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = {
            "answer": answer,
            "confidence": max(score for _, score in relevant_docs) if relevant_docs else 0.0
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "similarity_score": score
                }
                for text, score in relevant_docs
            ]
        
        return result
    
    # def clear_documents(self) -> None:
    #     """Clear all documents from the knowledge base"""
    #     self.chroma_client.delete_collection(self.collection.name)
    #     self.collection = self.chroma_client.create_collection(
    #         name=self.collection.name,
    #         metadata={"hnsw:space": "cosine"}
    #     )
    
    # def get_document_count(self) -> int:
    #     """Get the number of documents in the knowledge base"""
    #     return self.collection.count()