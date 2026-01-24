"""
Gateways module - Centralized interfaces for external API calls.
"""

from gateways.embedding_gateway import EmbeddingGateway
from gateways.llm_gateway import LLMGateway

__all__ = ['EmbeddingGateway', 'LLMGateway']
