"""
Embedding Gateway - Centralized interface for all embedding calls.

Simple wrapper around OpenAIEmbeddings with two core functions:
- embed_query: Embed a single query text
- embed_documents: Embed multiple documents
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSIONS
)


class EmbeddingGateway:
    """
    Centralized gateway for all embedding calls.
    
    Simple wrapper around OpenAIEmbeddings client.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = EMBEDDING_MODEL,
        base_url: str = EMBEDDING_BASE_URL,
        dimensions: int = EMBEDDING_DIMENSIONS
    ):
        """
        Initialize Embedding Gateway.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name (default from config)
            base_url: API base URL (default from config)
            dimensions: Embedding dimensions (default from config)
        """
        self.client = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        return self.client.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        return self.client.embed_documents(texts)
