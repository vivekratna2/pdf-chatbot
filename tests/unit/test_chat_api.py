import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import os

from src.api.chat_api import router, ChatRequest, ChatResponse


class TestChatAPI:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = TestClient(router)
    
    @patch('src.api.chat_api.RAGAgent')
    def test_ask_question_success(self, mock_rag_agent):
        """Test successful question asking"""
        # Setup mock
        mock_agent_instance = Mock()
        mock_agent_instance.process_query = AsyncMock(return_value={
            "answer": "This is the test answer",
            "confidence": 0.9,
            "sources": [{"text": "source1", "similarity_score": 0.9}],
            "query": "What is the test question?"
        })
        mock_rag_agent.return_value = mock_agent_instance
        
        # Test data
        request_data = {"query": "What is the test question?"}
        
        # Execute
        response = self.client.post("/ask", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["answer"] == "This is the test answer"
        assert response_data["confidence"] == 0.9
        assert response_data["query"] == "What is the test question?"
        assert len(response_data["sources"]) == 1
        
        # Verify RAGAgent was initialized correctly
        mock_rag_agent.assert_called_once()
        call_args = mock_rag_agent.call_args
        assert call_args[1]['embedding_model'] == os.getenv("MODEL_NAME", "mistral")
        assert call_args[1]['chat_model'] == os.getenv("MODEL_NAME", "mistral")
        assert call_args[1]['base_url'] == os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        
        # Verify process_query was called
        mock_agent_instance.process_query.assert_called_once_with("What is the test question?")
    
    @patch('src.api.chat_api.RAGAgent')
    def test_ask_question_with_environment_variables(self, mock_rag_agent):
        """Test ask question with custom environment variables"""
        # Setup mock
        mock_agent_instance = Mock()
        mock_agent_instance.process_query = AsyncMock(return_value={
            "answer": "Test answer",
            "confidence": 0.8,
            "sources": [],
            "query": "Test query"
        })
        mock_rag_agent.return_value = mock_agent_instance
        
        # Mock environment variables
        with patch.dict(os.environ, {
            'MODEL_NAME': 'llama2',
            'OLLAMA_BASE_URL': 'http://localhost:8080'
        }):
            # Test data
            request_data = {"query": "Test query"}
            
            # Execute
            response = self.client.post("/ask", json=request_data)
            
            # Assertions
            assert response.status_code == 200
            
            # Verify RAGAgent was initialized with environment variables
            call_args = mock_rag_agent.call_args
            assert call_args[1]['embedding_model'] == 'llama2'
            assert call_args[1]['chat_model'] == 'llama2'
            assert call_args[1]['base_url'] == 'http://localhost:8080'
    
    @patch('src.api.chat_api.OllamaRAG')
    def test_upload_file_success(self, mock_ollama_rag):
        """Test successful file upload"""
        # Setup mock
        mock_rag_instance = Mock()
        mock_rag_instance.add_documents.return_value = {"status": "success"}
        mock_ollama_rag.return_value = mock_rag_instance
        
        # Execute
        response = self.client.get("/upload_file")
        
        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["message"] == "PDF processed successfully"
        assert response_data["chroma response"] == {"status": "success"}
        
        # Verify OllamaRAG was initialized correctly
        mock_ollama_rag.assert_called_once()
        call_args = mock_ollama_rag.call_args
        assert call_args[1]['base_url'] == os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        
        # Verify add_documents was called with correct file path
        mock_rag_instance.add_documents.assert_called_once_with(file_path="/app/raw/Resume.pdf")
    
    @patch('src.api.chat_api.ChromaDBManager')
    def test_get_collection_count_success(self, mock_chromadb):
        """Test successful collection count retrieval"""
        # Setup mock
        mock_chromadb_instance = Mock()
        mock_chromadb_instance.get_collection_count.return_value = 42
        mock_chromadb.return_value = mock_chromadb_instance
        
        # Execute
        response = self.client.get("/collections/count")
        
        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["collection count"] == 42
        
        # Verify ChromaDBManager was initialized correctly
        mock_chromadb.assert_called_once_with(collection_name="resume_collection")
        
        # Verify get_collection_count was called
        mock_chromadb_instance.get_collection_count.assert_called_once()
    
    