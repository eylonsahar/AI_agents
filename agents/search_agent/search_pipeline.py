import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from typing import Dict, Any, List, Optional
import json

from agents.action_log import ActionLog
from agents.search_agent.vehicle_model_retriever import VehicleModelRetriever
from agents.search_agent.listings_retriever import ListingsRetriever
from agents.search_agent.decision_agent import DecisionAgent
from agents.utils.contracts import (
    RecommendationOutput, 
    UserQuery, 
    VehicleModelsResult, 
    PipelineResult,
    VehicleListing,
    ScoredListing
)
from gateways import EmbeddingGateway, LLMGateway
from config import NUM_OF_CHUNKS_TO_RETRIEVE, MAX_LISTINGS_PER_VEHICLE
import os


class SearchPipeline:
    """
    Orchestrates the complete vehicle search and recommendation pipeline.
    
    Pipeline stages:
    1. Vehicle Model Retrieval (vector search)
    2. Listings Retrieval (structured data filtering)
    3. Decision Ranking (scoring and final recommendations)
    """
    
    def __init__(
        self,
        pinecone_index,
        embedding_gateway: EmbeddingGateway,
        llm_gateway: LLMGateway,
        listings_csv_path: Optional[str] = None,
        action_log: Optional[ActionLog] = None
    ):
        self.vehicle_retriever = VehicleModelRetriever(
            pinecone_index=pinecone_index,
            embedding_gateway=embedding_gateway,
            llm_gateway=llm_gateway
        )
        self.listings_retriever = ListingsRetriever(csv_path=listings_csv_path, llm_gateway=llm_gateway)
        self.decision_agent = DecisionAgent()
        self.llm_gateway = llm_gateway
        self.action_log = action_log or ActionLog()
        
        # Store the recommended vehicle models from Stage 1
        self.recommended_models: Optional[VehicleModelsResult] = None
        
        # Store the retrieved listings from Stage 2 as VehicleListing objects
        self.retrieved_listings: List[VehicleListing] = []
        
        # Store the ranked listings from Stage 3
        self.ranked_listings: List[VehicleListing] = []
    
    def search(self, query: str, top_n_models: int = NUM_OF_CHUNKS_TO_RETRIEVE,
               year_min: Optional[int] = None, price_max: Optional[float] = None) -> PipelineResult:

        # Stage 1: Vehicle Model Retrieval
        vehicle_models_result, rag_details = self.vehicle_retriever.search_vehicle_models(
            query=query,
            top_n=top_n_models
        )
        
        # Log the LLM call for Stage 1
        self.action_log.add_step(
            module="SearchPipeline",
            submodule="VehicleModelRetrieval",
            prompt=rag_details.get("prompt", ""),
            response=rag_details.get("response", "")
        )
        
        vehicles = vehicle_models_result.get('vehicles', [])
        
        # Store the recommended models as a data model
        self.recommended_models = VehicleModelsResult.from_raw_result(
            query=query,
            raw_result=vehicle_models_result
        )
        
        # Stage 2: Listings Retrieval
        self.retrieved_listings = self.listings_retriever.retrieve_listings(
            vehicle_models_result=self.recommended_models,
            top_n=MAX_LISTINGS_PER_VEHICLE,
            year_min=year_min,
            price_max=price_max,
            user_query=query,
        )

        total_listings = len(self.retrieved_listings)
        
        # Stage 3: Decision Ranking (no LLM - just scoring logic)
        
        # Get scored listings once and use for both ranking and recommendations
        scored_listings = self.decision_agent.get_scored_listings(
            listings=self.retrieved_listings,
            vehicle_models_result=self.recommended_models
        )
        
        # Extract ranked listings from scored listings
        self.ranked_listings = [sl.listing for sl in scored_listings]
        
        return PipelineResult(
            query=query,
            vehicle_models_result=self.recommended_models,
            scored_listings=scored_listings,
            action_log=self.action_log
        )


def create_pipeline(
    pinecone_index,
    api_key: Optional[str] = None,
    embedding_gateway: Optional[EmbeddingGateway] = None,
    llm_gateway: Optional[LLMGateway] = None,
    listings_csv_path: Optional[str] = None
) -> SearchPipeline:
    """
    Factory function to create SearchPipeline instance.
    
    Args:
        pinecone_index: Pinecone index for vehicle models
        api_key: OpenAI API key (if not provided, will try to get from environment)
        embedding_gateway: Pre-configured EmbeddingGateway instance (optional)
        llm_gateway: Pre-configured LLMGateway instance (optional)
        listings_csv_path: Optional path to listings CSV file (defaults to config path)
        
    Returns:
        Configured SearchPipeline instance
    """
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
    
    # Create singleton gateways if not provided
    if embedding_gateway is None:
        embedding_gateway = EmbeddingGateway.get_instance(api_key=api_key)
    
    if llm_gateway is None:
        llm_gateway = LLMGateway.get_instance(api_key=api_key)
    
    return SearchPipeline(
        pinecone_index=pinecone_index,
        embedding_gateway=embedding_gateway,

        llm_gateway=llm_gateway,
        listings_csv_path=listings_csv_path
    )



if __name__ == "__main__":
    # Example usage (for testing)
    from agents.search_agent.rag_retrieval import get_pinecone_index
    
    # Get pinecone index
    pinecone_index = get_pinecone_index()
    
    # Create pipeline (will try to get API key from environment)
    pipeline = create_pipeline(pinecone_index=pinecone_index)
    
    # Test query
    query = "reliable mini for young couple"
    result = pipeline.search(query)
    
    # Print results
    print(json.dumps(result.to_dict(), indent=2))
    
    # Print action log
    result.print_action_log()
