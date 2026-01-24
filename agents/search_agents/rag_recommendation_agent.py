"""
RAG Recommendation Agent - Main Orchestrator.

Two-stage pipeline for vehicle discovery and ranking:
1. Stage 1: Vector search for vehicle models (VehicleModelRetriever)
2. Stage 2: Structured retrieval for listings (ListingsRetriever)
3. Generate reasoning using LLM Gateway
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from typing import Dict, Any, List
from agents.search_agents.vehicle_model_retriever import VehicleModelRetriever
from agents.search_agents.listings_retriever import ListingsRetriever
from gateways import LLMGateway
from agents.search_agents.contracts import UserQuery, RecommendationOutput
from agents.prompts import REASONING_SYSTEM_PROMPT


class RAGRecommendationAgent:
    """
    Main agent orchestrating the two-stage RAG pipeline for vehicle recommendations.
    
    Pipeline:
    User Query → Stage 1 (Vehicle Models) → Stage 2 (Listings) → Reasoning → Output
    """
    
    def __init__(
        self,
        vehicle_model_retriever: VehicleModelRetriever,
        listings_retriever: ListingsRetriever,
        llm_gateway: LLMGateway
    ):
        """
        Initialize the RAG Recommendation Agent.
        
        Args:
            vehicle_model_retriever: Stage 1 retriever
            listings_retriever: Stage 2 retriever
            llm_gateway: LLM gateway for reasoning generation
        """
        self.vehicle_model_retriever = vehicle_model_retriever
        self.listings_retriever = listings_retriever
        self.llm_gateway = llm_gateway
    
    def _generate_reasoning(
        self,
        query: str,
        listing: Any,
        vehicle_model: str
    ) -> str:
        """
        Generate reasoning for why a listing matches the user's query.
        
        Args:
            query: User's original query
            listing: VehicleListing object
            vehicle_model: Vehicle model name
        
        Returns:
            Reasoning text (max 2 sentences)
        """
        prompt = f"""{REASONING_SYSTEM_PROMPT}

User Query: {query}

Vehicle Listing:
- Model: {vehicle_model}
- Price: ${listing.price:,.0f}
- Year: {listing.year}
- Condition: {listing.condition or 'Not specified'}
- State: {listing.state or 'Not specified'}

Generate a brief explanation (max 2 sentences) of why this vehicle is a good match."""

        try:
            reasoning, _ = self.llm_gateway.call_llm(
                prompt=prompt,
                metadata={'listing_id': listing.listing_id}
            )
            return reasoning.strip()
        except Exception as e:
            # Fallback to template-based reasoning if LLM fails
            return f"This {listing.year} {vehicle_model} is available for ${listing.price:,.0f} in {listing.condition or 'good'} condition."
    
    def recommend(
        self,
        user_query: UserQuery,
        generate_reasoning: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the two-stage RAG pipeline and return recommendations.
        
        Args:
            user_query: UserQuery object with query text and parameters
            generate_reasoning: Whether to generate LLM-based reasoning
        
        Returns:
            Dictionary in the required format:
            {
              listing_id: {
                "score": float,
                "vehicle_model": str,
                "reasoning": str
              }
            }
        """
        # Stage 1: Retrieve vehicle models using vector search
        print(f"\n[Stage 1] Searching for vehicle models...")
        vehicle_models = self.vehicle_model_retriever.search_vehicle_models(
            query=user_query.query_text,
            top_n=user_query.top_n_models
        )
        print(f"Found {len(vehicle_models)} vehicle models")
        
        if not vehicle_models:
            print("No vehicle models found")
            return {}
        
        # Stage 2: Retrieve listings for the vehicle models
        print(f"\n[Stage 2] Retrieving listings...")
        listings = self.listings_retriever.retrieve_listings(
            vehicle_models=vehicle_models,
            top_k=user_query.top_k_listings,
            max_price=user_query.max_price,
            min_year=user_query.min_year,
            preferred_state=user_query.preferred_state
        )
        print(f"Found {len(listings)} listings")
        
        if not listings:
            print("No listings found")
            return {}
        
        # Generate output in required format
        print(f"\n[Stage 3] Generating recommendations...")
        recommendations = {}
        
        for listing in listings:
            # Generate reasoning
            if generate_reasoning:
                reasoning = self._generate_reasoning(
                    query=user_query.query_text,
                    listing=listing,
                    vehicle_model=listing.vehicle_model
                )
            else:
                reasoning = f"{listing.vehicle_model} - ${listing.price:,.0f}, {listing.year}"
            
            # Create recommendation output
            recommendation = RecommendationOutput(
                listing_id=listing.listing_id,
                score=listing.score,
                vehicle_model=listing.vehicle_model,
                reasoning=reasoning
            )
            
            recommendations[listing.listing_id] = recommendation.to_dict()
        
        return recommendations
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline execution.
        
        Returns:
            Dictionary with LLM usage stats
        """
        return {
            'llm_stats': self.llm_gateway.get_usage_stats()
        }
