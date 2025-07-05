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
    
    def generate_answer_with_chat_format(
        self,
        user_question: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate answer using chat format (alternative method)"""
        
        if system_prompt is None:
            system_prompt = """You are a helpful assistant. Use the provided context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly."""
        
        # Combine context
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"Context:\n{context_text}\n\nQuestion: {user_question}"
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.chat_stream_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                raise Exception(f"Failed to generate answer: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def generate_streaming_answer(
        self,
        user_question: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """Generate answer with streaming response"""
        
        if system_prompt is None:
            system_prompt = """You are a helpful assistant. Use the provided context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so clearly."""
        
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        full_prompt = f"""System: {system_prompt}

Context:
{context_text}

User Question: {user_question}

Answer:"""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                yield chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                raise Exception(f"Failed to generate streaming answer: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def check_model_availability(self) -> bool:
        """Check if the model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"].startswith(self.model_name) for model in models)
            return False
        except requests.exceptions.RequestException:
            return False