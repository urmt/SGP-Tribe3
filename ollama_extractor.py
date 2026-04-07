"""
Ollama Text Extractor for TRIBE v2
==================================
Replaces LLaMA-3.2-3B with Ollama embeddings for fast CPU text inference.
Uses an adapter MLP to map Ollama embeddings to LLaMA-like embeddings.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from neuralset.extractors.text import BaseText
from neuralset.extractors.base import BaseExtractor


class OllamaTextExtractor(BaseText):
    """Text extractor that uses Ollama embeddings instead of LLaMA.
    
    Produces embeddings in the same format as HuggingFaceText:
    - Per-word embeddings of shape (3, 3072)
    - Uses adapter MLP to map Ollama embeddings (4096d) → LLaMA-like (9216d)
    """
    
    event_types: tuple = ("Word",)
    
    def __init__(
        self,
        model_name: str = "mistral",
        ollama_host: str = "http://localhost:11434",
        adapter_path: Optional[str] = None,
        target_dim: int = 9216,
        ollama_dim: int = 4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.target_dim = target_dim
        self.ollama_dim = ollama_dim
        
        # Initialize Ollama client
        if OLLAMA_AVAILABLE:
            self.client = ollama.Client(host=ollama_host)
        else:
            self.client = None
            
        # Load adapter if available
        self.adapter = None
        if adapter_path and os.path.exists(adapter_path):
            self.adapter = EmbeddingAdapter(ollama_dim, target_dim)
            self.adapter.load_state_dict(torch.load(adapter_path, map_location='cpu', weights_only=True))
            self.adapter.eval()
            print(f"[OllamaTextExtractor] Loaded adapter from {adapter_path}")
        else:
            print(f"[OllamaTextExtractor] No adapter found at {adapter_path}, using zero-pad fallback")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using Ollama.
        
        Returns:
            numpy array of shape (target_dim,) or (ollama_dim,) if no adapter
        """
        if self.client is None:
            raise RuntimeError("Ollama client not available. Install with: pip install ollama")
        
        try:
            response = self.client.embed(
                model=self.model_name,
                input=text,
                truncate=True
            )
            embedding = np.array(response['embeddings'][0], dtype=np.float32)
            
            # Apply adapter if available
            if self.adapter is not None:
                with torch.no_grad():
                    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)
                    adapted = self.adapter(embedding_tensor).squeeze(0).numpy()
                return adapted
            
            # Fallback: zero-pad to target dimension
            if len(embedding) < self.target_dim:
                padded = np.zeros(self.target_dim, dtype=np.float32)
                padded[:len(embedding)] = embedding
                return padded
            
            return embedding[:self.target_dim]
            
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts.
        
        Returns:
            numpy array of shape (n_texts, target_dim)
        """
        if self.client is None:
            raise RuntimeError("Ollama client not available")
        
        try:
            response = self.client.embed(
                model=self.model_name,
                input=texts,
                truncate=True
            )
            embeddings = np.array(response['embeddings'], dtype=np.float32)
            
            # Apply adapter if available
            if self.adapter is not None:
                with torch.no_grad():
                    embedding_tensor = torch.from_numpy(embeddings)
                    adapted = self.adapter(embedding_tensor).numpy()
                return adapted
            
            # Fallback: zero-pad
            if embeddings.shape[1] < self.target_dim:
                padded = np.zeros((len(texts), self.target_dim), dtype=np.float32)
                padded[:, :embeddings.shape[1]] = embeddings
                return padded
            
            return embeddings[:, :self.target_dim]
            
        except Exception as e:
            raise RuntimeError(f"Ollama batch embedding failed: {e}")


class EmbeddingAdapter(nn.Module):
    """MLP adapter to map Ollama embeddings to LLaMA-like embeddings.
    
    Architecture: ollama_dim → hidden → target_dim
    """
    
    def __init__(self, input_dim: int = 4096, target_dim: int = 9216, hidden_dim: int = 8192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )
    
    def forward(self, x):
        return self.network(x)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[EmbeddingAdapter] Saved to {path}")
    
    @classmethod
    def load(cls, path: str, input_dim: int = 4096, target_dim: int = 9216):
        adapter = cls(input_dim, target_dim)
        adapter.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        adapter.eval()
        print(f"[EmbeddingAdapter] Loaded from {path}")
        return adapter


def check_ollama_connection(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    if not OLLAMA_AVAILABLE:
        return False
    
    try:
        client = ollama.Client(host=host)
        response = client.list()
        return True
    except Exception:
        return False


def get_available_ollama_models(host: str = "http://localhost:11434") -> List[str]:
    """Get list of available Ollama models."""
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        client = ollama.Client(host=host)
        response = client.list()
        return [m['name'] for m in response['models']]
    except Exception:
        return []
