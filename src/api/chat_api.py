import logging
from fastapi import APIRouter, HTTPException, status, Body, Depends

from src.core.chromadb_manager import ChromaDBManager
from src.core.ollama_embedding import OllamaEmbedding
from src.utils.file_chunker import PDFChunker
from src.core.ollama_rag import OllamaRAG

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/ask")
async def ask_question():
    """
    Endpoint to ask a question.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        pdfchunker = PDFChunker()
        ollama_embedding = OllamaEmbedding()
        chromadb = ChromaDBManager(collection_name="resume_collection")
        chunks = pdfchunker.process_pdf("/app/raw/Resume.pdf")
        # embeddings = ollama_embedding.embed_documents(chunks)
        response = chromadb.add_documents(documents=chunks)
        return {"message": "PDF processed successfully", "chroma response": response}
        # return {"message": "This is a placeholder for the ask question endpoint."}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@router.get("/upload_file")
async def upload_file():
    """
    Endpoint to ask a question.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        pdfchunker = PDFChunker()
        ollama_embedding = OllamaEmbedding()
        chromadb = ChromaDBManager(collection_name="resume_collection")
        chunks = pdfchunker.process_pdf("/app/raw/Resume.pdf")
        # embeddings = ollama_embedding.embed_documents(chunks)
        response = chromadb.add_documents(documents=chunks)
        return {"message": "PDF processed successfully", "chroma response": response}
        # return {"message": "This is a placeholder for the ask question endpoint."}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@router.post("/ask")
async def query_chroma(
    question: str = Body(..., description="The question to ask the ChromaDB collection.")
):
    """
    Endpoint to query the ChromaDB collection.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        chromadb = ChromaDBManager(collection_name="resume_collection")
        query_text = "What is the candidate's name?"
        results = chromadb.query(query_text=query_text)
        return {"message": "Query executed successfully", "results": results}
    except Exception as e:
        logger.error(f"Error in query_chroma: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/user_ask")
async def user_ask():
    """
    Endpoint to handle user questions.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        # Initialize RAG system
        rag = OllamaRAG(
            embedding_model="mistral",
            chat_model="mistral",
            base_url="http://ollama:11434",
            top_k=3
        )
        
        question = "Who is vivek?"
        result = rag.generate_answer(question)
        
        return {
            "message": "This is a placeholder for the user ask endpoint.",
            "question": question,
            "answer": result
            }
    except Exception as e:
        logger.error(f"Error in user_ask: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
