"""
Gateways module - Centralized interfaces for external API calls.

Both EmbeddingGateway and LLMGateway are implemented as singletons.
Use get_embedding_gateway() and get_llm_gateway() for convenient access.
"""

from gateways.embedding_gateway import EmbeddingGateway
from gateways.llm_gateway import LLMGateway

# Convenience functions for getting singleton instances
def get_embedding_gateway(api_key: str, **kwargs) -> EmbeddingGateway:
    """
    Get the singleton instance of EmbeddingGateway.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional arguments to pass to EmbeddingGateway
    
    Returns:
        The singleton EmbeddingGateway instance
    """
    return EmbeddingGateway.get_instance(api_key=api_key, **kwargs)

def get_llm_gateway(api_key: str, **kwargs) -> LLMGateway:
    """
    Get the singleton instance of LLMGateway.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional arguments to pass to LLMGateway
    
    Returns:
        The singleton LLMGateway instance
    """
    return LLMGateway.get_instance(api_key=api_key, **kwargs)

__all__ = [
    'EmbeddingGateway', 
    'LLMGateway',
    'get_embedding_gateway',
    'get_llm_gateway'
]
