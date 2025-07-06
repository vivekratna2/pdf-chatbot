from typing import List
import numpy as np
import requests
from sklearn.preprocessing import normalize


class OllamaEmbedding:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://ollama:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.embed_url = f"{self.base_url}/api/embeddings"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        for text in texts:
            response = requests.post(
                self.embed_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                
                # Normalize the embedding vector
                embedding = np.array(embedding)
                normalized_embedding = normalize([embedding], norm='l2')[0]
                
                embeddings.append(normalized_embedding.tolist())
            else:
                raise Exception(f"Failed to get embedding: {response.text}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = requests.post(
            self.embed_url,
            json={
                "model": self.model_name,
                "prompt": text
            }
        )
        if response.status_code == 200:
            embedding = response.json()["embedding"]
            
            # Normalize the embedding vector
            embedding = np.array(embedding)
            normalized_embedding = normalize([embedding], norm='l2')[0]
            
            return normalized_embedding.tolist()
        else:
            raise Exception(f"Failed to get embedding: {response.text}")
