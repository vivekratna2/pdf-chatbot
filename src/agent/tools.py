from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import asyncio

from src.core.chromadb_manager import ChromaDBManager
from src.utils.file_chunker import PDFChunker


@tool
def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for documents in the knowledge base"""
    chromadb = ChromaDBManager(collection_name="resume_collection")
    results = chromadb.query(query_text=query, n_results=top_k)
    
    documents = []
    if results['documents'] and results['distances']:
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            documents.append({
                "content": doc,
                "similarity": 1 - distance,
                "metadata": {}
            })
    
    return documents


@tool
def process_pdf_document(file_path: str) -> Dict[str, Any]:
    """Process a PDF document and add it to the knowledge base"""
    try:
        pdf_chunker = PDFChunker()
        chromadb = ChromaDBManager(collection_name="resume_collection")
        
        chunks = pdf_chunker.process_pdf(file_path)
        if chunks:
            chromadb.add_documents(documents=chunks)
            return {
                "success": True,
                "message": f"Successfully processed {len(chunks)} chunks",
                "chunk_count": len(chunks)
            }
        else:
            return {
                "success": False,
                "message": "No content extracted from PDF",
                "chunk_count": 0
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}",
            "chunk_count": 0
        }


@tool
def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the document collection"""
    chromadb = ChromaDBManager(collection_name="resume_collection")
    count = chromadb.get_collection_count()
    
    return {
        "document_count": count,
        "collection_name": "resume_collection",
        "status": "active" if count > 0 else "empty"
    }