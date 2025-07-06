import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from src.core.ollama_embedding import OllamaEmbedding


class TestOllamaEmbedding:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.embedding_client = OllamaEmbedding(
            model_name="mistral",
            base_url="http://ollama:11434"
        )
    
    @patch('src.core.ollama_embedding.requests.post')
    def test_embed_documents_success(self, mock_post):
        """Test successful embedding of multiple documents"""
        # Setup mock responses
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
        
        mock_post.side_effect = [mock_response1, mock_response2]
        
        # Test data
        texts = ["Hello world", "How are you?"]
        
        # Execute
        result = self.embedding_client.embed_documents(texts)
        
        # Assertions
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        # Verify requests were made correctly
        assert mock_post.call_count == 2
        
        # Check first call
        first_call = mock_post.call_args_list[0]
        assert first_call[0][0] == "http://ollama:11434/api/embeddings"
        assert first_call[1]['json'] == {
            "model": "mistral",
            "prompt": "Hello world"
        }
        
        # Check second call
        second_call = mock_post.call_args_list[1]
        assert second_call[0][0] == "http://ollama:11434/api/embeddings"
        assert second_call[1]['json'] == {
            "model": "mistral",
            "prompt": "How are you?"
        }
    
    @patch('src.core.ollama_embedding.requests.post')
    def test_embed_documents_empty_list(self, mock_post):
        """Test embedding empty list of documents"""
        # Execute
        result = self.embedding_client.embed_documents([])
        
        # Assertions
        assert result == []
        mock_post.assert_not_called()
    
    @patch('src.core.ollama_embedding.requests.post')
    def test_embed_documents_http_error(self, mock_post):
        """Test handling of HTTP errors"""
        # Setup mock response with error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # Test data
        texts = ["Test document"]
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            self.embedding_client.embed_documents(texts)
        
        assert "Failed to get embedding: Internal Server Error" in str(exc_info.value)
        mock_post.assert_called_once()
    
    @patch('src.core.ollama_embedding.requests.post')
    def test_embed_documents_missing_embedding_key(self, mock_post):
        """Test handling when response is missing embedding key"""
        # Setup mock response without embedding key
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "No embedding key"}
        mock_post.return_value = mock_response
        
        # Test data
        texts = ["Test document"]
        
        # Execute and assert exception
        with pytest.raises(KeyError):
            self.embedding_client.embed_documents(texts)
        
        mock_post.assert_called_once()
    
    @patch('src.core.ollama_embedding.requests.post')
    def test_embed_query_success(self, mock_post):
        """Test successful embedding of a single query"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        mock_post.return_value = mock_response
        
        # Test data
        query_text = "What is the weather like today?"
        
        # Execute
        result = self.embedding_client.embed_query(query_text)
        
        # Assertions
        assert result == [0.1, 0.2, 0.3, 0.4]
        
        # Verify request was made correctly
        mock_post.assert_called_once_with(
            "http://ollama:11434/api/embeddings",
            json={
                "model": "mistral",
                "prompt": "What is the weather like today?"
            }
        )
    
    @patch('src.core.ollama_embedding.requests.post')
    def test_embed_query_empty_string(self, mock_post):
        """Test embedding of empty string query"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.0, 0.0, 0.0]}
        mock_post.return_value = mock_response
        
        # Test data
        query_text = ""
        
        # Execute
        result = self.embedding_client.embed_query(query_text)
        
        # Assertions
        assert result == [0.0, 0.0, 0.0]
        
        # Verify request was made with empty string
        mock_post.assert_called_once_with(
            "http://ollama:11434/api/embeddings",
            json={
                "model": "mistral",
                "prompt": ""
            }
        )
    