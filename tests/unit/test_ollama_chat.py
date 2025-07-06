import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from src.core.ollama_chat import OllamaChat


class TestOllamaChat:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.chat_client = OllamaChat(
            model_name="mistral",
            base_url="http://ollama:11434"
        )
    
    @patch('src.core.ollama_chat.requests.post')
    def test_generate_answer_success(self, mock_post):
        """Test successful answer generation"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test answer."}
        mock_post.return_value = mock_response
        
        # Test data
        user_question = "What is the weather like?"
        context = ["It's sunny today", "Temperature is 25°C"]
        
        # Execute
        result = self.chat_client.generate_answer(user_question, context)
        
        # Assertions
        assert result == "This is a test answer."
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check URL
        assert call_args[0][0] == "http://ollama:11434/api/generate"
        
        # Check payload structure
        payload = call_args[1]['json']
        assert payload['model'] == "mistral"
        assert payload['stream'] is False
        assert payload['options']['temperature'] == 0.7
        assert payload['options']['num_predict'] == 1000
        assert payload['keep_alive'] == "15m"
        
        # Check prompt contains user question and context
        assert "What is the weather like?" in payload['prompt']
        assert "Context 1: It's sunny today" in payload['prompt']
        assert "Context 2: Temperature is 25°C" in payload['prompt']
    