"""
LLM Gateway - Centralized interface for all LLM calls.

This module provides a single entry point for LLM interactions with:
- Retry logic with exponential backoff
- Timeout handling
- Token usage tracking
- Model selection

Wraps the existing ChatOpenAI client from the RAG system.
Implemented as a singleton to ensure only one instance exists.
"""

import time
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from config import (
    CHAT_MODEL,
    CHAT_BASE_URL
)

# Default LLM settings
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY = 1.0
LLM_TIMEOUT = 120  # Increased for reasoning models that need more processing time
LLM_TEMPERATURE = 1.0
LLM_MAX_TOKENS = 4000  # Increased to allow for reasoning tokens + output content



class LLMGateway:
    """
    Centralized gateway for all LLM calls (Singleton).
    
    Provides retry logic, timeout handling, and usage tracking.
    Reuses the existing ChatOpenAI client from the RAG system.
    Only one instance of this class will exist throughout the application.
    """
    
    _instance: Optional['LLMGateway'] = None
    
    def __init__(
        self,
        api_key: str,
        model: str = CHAT_MODEL,
        base_url: str = CHAT_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_TIMEOUT
    ):
        """
        Initialize LLM Gateway.
        
        Note: Use get_instance() class method instead of direct instantiation
        to ensure singleton behavior.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default from config)
            base_url: API base URL (default from config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.model = model
        self.timeout = timeout
        
        # Initialize ChatOpenAI client (reuses existing RAG logic)
        # Create a custom client that filters out unsupported parameters
        from langchain_openai import ChatOpenAI
        
        class CustomChatOpenAI(ChatOpenAI):
            """Custom ChatOpenAI that filters out unsupported parameters."""
            
            def _stream(self, messages, stop=None, **kwargs):
                # Remove the stop parameter before calling parent
                return super()._stream(messages, stop=None, **kwargs)
            
            def _generate(self, messages, stop=None, **kwargs):
                # Remove the stop parameter before calling parent
                return super()._generate(messages, stop=None, **kwargs)
        
        self.client = CustomChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout
        )

    @classmethod
    def get_instance(
        cls,
        api_key: str,
        model: str = CHAT_MODEL,
        base_url: str = CHAT_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_TIMEOUT
    ) -> 'LLMGateway':
        """
        Get the singleton instance of LLMGateway.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default from config)
            base_url: API base URL (default from config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        
        Returns:
            The singleton LLMGateway instance
        """
        if cls._instance is None:
            cls._instance = cls(
                api_key=api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def call_llm(
        self,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        retry_attempts: int = LLM_RETRY_ATTEMPTS,
        retry_delay: float = LLM_RETRY_DELAY
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call the LLM with retry logic and error handling.
        
        Args:
            prompt: Input prompt for the LLM
            metadata: Optional metadata for logging/tracking
            retry_attempts: Number of retry attempts on failure
            retry_delay: Initial delay between retries (exponential backoff)
        
        Returns:
            Tuple of (response_text, usage_stats)
            
        Raises:
            Exception: If all retry attempts fail
        """
        metadata = metadata or {}
        last_exception = None
        
        for attempt in range(retry_attempts):
            try:
                # Invoke the LLM using ChatOpenAI pattern
                response = self.client.invoke(prompt)
                
                # Extract response content
                response_text = response.content
                
                # Extract usage statistics if available
                usage = {
                    'prompt_tokens': response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0),
                    'completion_tokens': response.response_metadata.get('token_usage', {}).get('completion_tokens', 0),
                    'total_tokens': response.response_metadata.get('token_usage', {}).get('total_tokens', 0),
                    'model': self.model,
                    'metadata': metadata
                }
                
                return response_text, usage
                
            except Exception as e:
                last_exception = e
                
                if attempt < retry_attempts - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"LLM call failed (attempt {attempt + 1}/{retry_attempts}): {str(e)}")
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    print(f"LLM call failed after {retry_attempts} attempts")
        
        # All retries exhausted
        raise Exception(f"LLM Gateway: All {retry_attempts} attempts failed. Last error: {str(last_exception)}")
    