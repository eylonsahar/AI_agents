import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from typing import List, Dict, Any
from pydantic import ValidationError
from agents.utils.contracts import VehicleModel
from agents.utils.response_models import VehicleRecommendationResponse
from agents.search_agent.rag_retrieval import RAGRetriever
from gateways import EmbeddingGateway, LLMGateway
from config import NUM_OF_CHUNKS_TO_RETRIEVE
from agents.prompts import VEHICLE_MODEL_SYSTEM_PROMPT


class VehicleModelRetriever:
    """
    Retrieves vehicle models using vector similarity search.
    
    Stage 1 of the two-stage RAG pipeline.
    Uses RAGRetriever instance to perform all embedding and search operations.
    """
    
    def __init__(self, pinecone_index, embedding_gateway: EmbeddingGateway, llm_gateway: LLMGateway):
        """
        Initialize the vehicle model retriever.
        
        Args:
            pinecone_index: Pinecone index object for vehicle models (filtered-vehicles-info)
            embedding_gateway: EmbeddingGateway instance
            llm_gateway: LLMGateway instance for generating vehicle recommendations
        """
        self.rag_retriever = RAGRetriever(
            pinecone_index=pinecone_index,
            embedding_gateway=embedding_gateway,
            llm_gateway=llm_gateway,
            system_prompt=VEHICLE_MODEL_SYSTEM_PROMPT,
            context_formatter=VehicleModelRetriever.format_vehicle_context
        )
        self.llm_gateway = llm_gateway
    
    def search_vehicle_models(
        self,
        query: str,
        top_n: int = NUM_OF_CHUNKS_TO_RETRIEVE
    ) -> Dict[str, Any]:
        """
        Search for vehicle models based on user needs using vector similarity.
        Uses Pydantic validation to ensure correct response format.
        
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
        
        # Simple validation without retry logic
        validated_response = self._validate_response(
            response_text=rag_result.get('response', ''),
            query=query,
            context=rag_result.get('chunks', [])
        )
        
        return validated_response
    
    def _validate_response(
        self,
        response_text: str,
        query: str,
        context: List[Dict]
    ) -> Dict[str, Any]:
        """
        Validate LLM response with Pydantic without retry logic.
        
        Args:
            response_text: Raw response from LLM
            query: Original user query
            context: Retrieved context chunks
        
        Returns:
            Validated response dictionary or fallback response
        """
        try:
            # Parse JSON
            parsed_response = json.loads(response_text)
            
            # Validate with Pydantic
            validated = VehicleRecommendationResponse(**parsed_response)
            
            # Return as dictionary
            return validated.model_dump()
        
        except (json.JSONDecodeError, ValidationError) as e:
            # Return fallback response on validation failure
            return {
                'vehicles': [],
                'explanation': f"Failed to validate response: {str(e)}"
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

