from typing import List, Dict, Any, Optional
import requests
import json


class OllamaChat:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://ollama:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.chat_url = f"{self.base_url}/api/generate"
        self.chat_stream_url = f"{self.base_url}/api/chat"
    
    def generate_answer(
        self, 
        user_question: str, 
        context: List[str], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate an answer based on user question and context"""
        
        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = """You are a helpful assistant. Use the provided context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly."""
        
        # Combine context into a single string
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        # Create the full prompt
        full_prompt = f"""System: {system_prompt}

Context:
{context_text}

User Question: {user_question}

Answer:"""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "keep_alive": "15m"  # Keep model loaded for 10 minutes after last use
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Failed to generate answer: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    