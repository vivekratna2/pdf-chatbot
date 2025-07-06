import logging
import os
from pathlib import Path
import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Body, Depends
from pydantic import BaseModel

from src.agent.langgraph_agent import RAGAgent
from src.core.chromadb_manager import ChromaDBManager
from src.core.ollama_embedding import OllamaEmbedding
from src.utils.file_chunker import PDFChunker
from src.core.ollama_rag import OllamaRAG

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: list
    query: str


@router.post("/ask")
async def ask_question(request: ChatRequest):
    """
    Endpoint to ask a question.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        rag_agent = RAGAgent(embedding_model=os.getenv("MODEL_NAME", "mistral"),
                     chat_model=os.getenv("MODEL_NAME", "mistral"),
                     base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
        result = await rag_agent.process_query(request.query)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.post("/file_upload")
async def file_upload(file: UploadFile = File(...)):
    """
    Endpoint to upload a file to the /raw folder.
    """
    try:
        # Define the raw folder path
        raw_folder = Path("/app/raw")
        
        # Create the raw folder if it doesn't exist
        raw_folder.mkdir(parents=True, exist_ok=True)
        
        # Define the file path
        file_path = raw_folder / file.filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        ollama_rag = OllamaRAG(base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
        response = ollama_rag.add_documents(file_path=file_path)
        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "file_path": str(file_path),
            "chroma response": response
        }
    except Exception as e:
        logger.error(f"Error in file_upload: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/upload_file")
async def upload_file():
    """
    Endpoint to ask a question.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        ollama_rag = OllamaRAG(base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
        response = ollama_rag.add_documents(file_path="/app/raw/Resume.pdf")
        return {"message": "PDF processed successfully", "chroma response": response}
        # return {"message": "This is a placeholder for the ask question endpoint."}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    
@router.get("/test")
async def user_ask():
    """
    Endpoint to handle user questions.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        # Initialize RAG system
        # rag = OllamaRAG(
        #     embedding_model=os.getenv("MODEL_NAME", "mistral"),
        #     chat_model=os.getenv("MODEL_NAME", "mistral"),
        #     base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        #     top_k=3
        # )
        
        # question = "Who is vivek?"
        # result = rag.generate_answer(question)
        
        # return {
        #     "message": "This is a placeholder for the user ask endpoint.",
        #     "question": question,
        #     "answer": result
        #     }
        c = ChromaDBManager(collection_name = "resume_collection")
        result = c.query(query_text="Who is vivek?")
        return {
            "message": "This is a placeholder for the user ask endpoint.",
            "query": "Who is sangam?",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in user_ask: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    
@router.get("/collections/count")
async def get_collection_count():
    """
    Endpoint to get the number of documents in a specific collection.
    """
    try:
        chromadb = ChromaDBManager(collection_name="resume_collection")
        count = chromadb.get_collection_count()
        return {"collection count": count}
    except Exception as e:
        logger.error(f"Error in get_collection_count: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    
@router.delete("/collections/reset")
async def reset_collection():
    """
    Endpoint to delete a specific collection.
    """
    try:
        chromadb = ChromaDBManager(collection_name="resume_collection")
        chromadb.reset_collection()
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error in delete_collection: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
