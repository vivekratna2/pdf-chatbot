import logging
import os
from fastapi import APIRouter, HTTPException, status, Body, Depends
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
    temperature: float = 0.7
    max_tokens: int = 1000

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
    

# @router.post("/ask")
# async def query_chroma(
#     question: str = Body(..., description="The question to ask the ChromaDB collection.")
# ):
#     """
#     Endpoint to query the ChromaDB collection.
#     This is a placeholder endpoint that can be extended later.
#     """
#     try:
#         chromadb = ChromaDBManager(collection_name="resume_collection")
#         query_text = "What is the candidate's name?"
#         results = chromadb.query(query_text=query_text)
#         return {"message": "Query executed successfully", "results": results}
#     except Exception as e:
#         logger.error(f"Error in query_chroma: {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/user_ask")
async def user_ask():
    """
    Endpoint to handle user questions.
    This is a placeholder endpoint that can be extended later.
    """
    try:
        # Initialize RAG system
        rag = OllamaRAG(
            embedding_model=os.getenv("MODEL_NAME", "mistral"),
            chat_model=os.getenv("MODEL_NAME", "mistral"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
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
