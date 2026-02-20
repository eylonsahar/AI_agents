"""
Embedding Gateway - Centralized interface for all embedding calls.

Simple wrapper around OpenAIEmbeddings with two core functions:
- embed_query: Embed a single query text
- embed_documents: Embed multiple documents

Implemented as a singleton to ensure only one instance exists.
"""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSIONS
)


class EmbeddingGateway:
    """
    Centralized gateway for all embedding calls (Singleton).
    
    Simple wrapper around OpenAIEmbeddings client.
    Only one instance of this class will exist throughout the application.
    """
    
    _instance: Optional['EmbeddingGateway'] = None
    
    def __init__(
        self,
        api_key: str,
        model: str = EMBEDDING_MODEL,
        base_url: str = EMBEDDING_BASE_URL,
        dimensions: int = EMBEDDING_DIMENSIONS
    ):
        """
        Initialize Embedding Gateway.
        
        Note: Use get_instance() class method instead of direct instantiation
        to ensure singleton behavior.
        
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
    
    @classmethod
    def get_instance(
        cls,
        api_key: str,
        model: str = EMBEDDING_MODEL,
        base_url: str = EMBEDDING_BASE_URL,
        dimensions: int = EMBEDDING_DIMENSIONS
    ) -> 'EmbeddingGateway':
        """
        Get the singleton instance of EmbeddingGateway.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name (default from config)
            base_url: API base URL (default from config)
            dimensions: Embedding dimensions (default from config)
        
        Returns:
            The singleton EmbeddingGateway instance
        """
        if cls._instance is None:
            cls._instance = cls(
                api_key=api_key,
                model=model,
                base_url=base_url,
                dimensions=dimensions
            )
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
    
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