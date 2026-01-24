"""
Vehicle Model Retriever - Stage 1 of RAG Recommendation Pipeline.

Picks vehicle models based on user needs using the vehicles-mini-lm index.
Delegates all embedding and search operations to RAGRetriever.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from typing import List, Dict, Any
from pydantic import ValidationError
from agents.search_agents.contracts import VehicleModel
from agents.search_agents.response_models import VehicleRecommendationResponse
from rag.src.rag_retrieval import RAGRetriever
from gateways import EmbeddingGateway, LLMGateway
from config import TOP_K
from agents.prompts import VEHICLE_MODEL_SYSTEM_PROMPT


class VehicleModelRetriever:
    """
    Retrieves vehicle models using vector similarity search.
    
    Stage 1 of the two-stage RAG pipeline.
    Uses RAGRetriever instance to perform all embedding and search operations.
    """
    
    def __init__(self, pinecone_index, embedding_gateway: EmbeddingGateway, llm_gateway: LLMGateway, max_retries: int = 3):
        """
        Initialize the vehicle model retriever.
        
        Args:
            pinecone_index: Pinecone index object for vehicle models (vehicles-mini-lm)
            embedding_gateway: EmbeddingGateway instance
            llm_gateway: LLMGateway instance for generating vehicle recommendations
            max_retries: Maximum number of retry attempts for validation failures
        """
        self.rag_retriever = RAGRetriever(
            pinecone_index=pinecone_index,
            embedding_gateway=embedding_gateway,
            llm_gateway=llm_gateway,
            system_prompt=VEHICLE_MODEL_SYSTEM_PROMPT,
            context_formatter=VehicleModelRetriever.format_vehicle_context
        )
        self.llm_gateway = llm_gateway
        self.max_retries = max_retries
    
    def search_vehicle_models(
        self,
        query: str,
        top_n: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Search for vehicle models based on user needs using vector similarity.
        Uses Pydantic validation with retry logic to ensure correct response format.
        
        Args:
            query: User's natural language query describing their needs 
                   (e.g., "reliable SUV for family", "fuel-efficient sedan")
            top_n: Number of vehicle models to retrieve
        
        Returns:
            Dictionary with structure:
            {
                "vehicles": List of vehicle dictionaries with make, model, body_type, years, match_score, match_reason
                "explanation": Overall reasoning for the selections
            }
        """
        rag_result = self.rag_retriever.query(query, top_k=top_n)
        
        # Try to validate and parse the response with retries
        validated_response = self._validate_with_retry(
            response_text=rag_result.get('response', ''),
            query=query,
            context=rag_result.get('chunks', [])
        )
        
        return validated_response
    
    def _validate_with_retry(
        self,
        response_text: str,
        query: str,
        context: List[Dict],
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        Validate LLM response with Pydantic and retry if validation fails.
        
        Args:
            response_text: Raw response from LLM
            query: Original user query
            context: Retrieved context chunks
            attempt: Current attempt number
        
        Returns:
            Validated response dictionary
        """
        try:
            # Parse JSON
            parsed_response = json.loads(response_text)
            
            # Validate with Pydantic
            validated = VehicleRecommendationResponse(**parsed_response)
            
            # Return as dictionary
            return validated.model_dump()
        
        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = f"Validation error (attempt {attempt}/{self.max_retries}): {str(e)}"
            print(error_msg)
            
            # If we haven't exceeded max retries, try again
            if attempt < self.max_retries:
                print(f"Retrying with corrected prompt...")
                
                # Format context for retry
                formatted_context = VehicleModelRetriever.format_vehicle_context(context)
                
                # Create a corrected prompt with error feedback
                retry_prompt = f"""{VEHICLE_MODEL_SYSTEM_PROMPT}

                    PREVIOUS ATTEMPT FAILED:
                    Error: {str(e)}
                    
                    Please ensure you return ONLY a valid JSON object with the exact structure specified.
                    
                    Context:
                    {formatted_context}
                    
                    Question: {query}"""
                
                # Call LLM directly for retry
                retry_response, _ = self.llm_gateway.call_llm(prompt=retry_prompt)
                
                # Recursive retry
                return self._validate_with_retry(
                    response_text=retry_response,
                    query=query,
                    context=context,
                    attempt=attempt + 1
                )
            
            # All retries exhausted, return fallback
            return {
                'vehicles': [],
                'explanation': f"Failed to get valid response after {self.max_retries} attempts. Last error: {str(e)}"
            }

    @staticmethod
    def format_vehicle_context(chunks: List[Dict]) -> str:
        """
        Vehicle models specific context formatter.
        Formats retrieved vehicle data for the LLM to analyze.

        Args:
            chunks: List of chunks with metadata

        Returns:
            Formatted context string for vehicle models
        """
        if not chunks:
            return "No vehicle data retrieved."
        
        context_parts = ["Retrieved Vehicle Data:\n"]

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})

            context_parts.append(f"Vehicle {i}:")
            context_parts.append(f"  Make: {metadata.get('make', 'N/A')}")
            context_parts.append(f"  Model: {metadata.get('model', 'N/A')}")
            context_parts.append(f"  Body Type: {metadata.get('body_type', 'N/A')}")
            context_parts.append(f"  Years: {metadata.get('years', 'N/A')}")
            context_parts.append(f"  Description: {metadata.get('all_text', 'N/A')}")
            context_parts.append(f"  Similarity Score: {chunk.get('score', 0):.3f}")
            context_parts.append("")

        return "\n".join(context_parts)

