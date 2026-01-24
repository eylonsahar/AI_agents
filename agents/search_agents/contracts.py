"""
Data contracts for the agent-based RAG system.

Defines explicit data structures for vehicle models, listings, and recommendations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class VehicleModel:
    """
    Represents a vehicle model retrieved from Stage 1 (vector search).
    """
    model_id: str
    make: str
    model: str
    body_type: str
    years: str
    score: float  # Similarity score from vector search
    metadata: Dict[str, Any]  # Additional metadata (url, all_text, etc.)


@dataclass
class VehicleListing:
    """
    Represents a specific vehicle listing from Stage 2 (structured retrieval).
    """
    listing_id: str
    vehicle_model: str  # e.g., "Ford Focus"
    manufacturer: str
    model: str
    price: float
    year: int
    condition: Optional[str]
    paint_color: Optional[str]
    state: Optional[str]
    region: Optional[str]
    score: float  # Computed score based on filtering criteria


@dataclass
class RecommendationOutput:
    """
    Final output format for a single recommendation.
    
    This is the contract specified in the requirements:
    {
      listing_id: {
        "score": float,
        "vehicle_model": str,
        "reasoning": str
      }
    }
    """
    listing_id: str
    score: float
    vehicle_model: str
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to the required output format."""
        return {
            "score": self.score,
            "vehicle_model": self.vehicle_model,
            "reasoning": self.reasoning
        }


@dataclass
class UserQuery:
    """
    User input for vehicle recommendation.
    
    TODO: Extend with additional fields based on user requirements:
    - budget_min, budget_max
    - preferred_location
    - preferred_condition
    - preferred_years
    """
    query_text: str  # Natural language query
    top_n_models: int = 5  # Number of models to retrieve in Stage 1
    top_k_listings: int = 10  # Number of listings to return in Stage 2
    
    # Optional filters (TODO: expand based on user needs)
    max_price: Optional[float] = None
    min_year: Optional[int] = None
    preferred_state: Optional[str] = None
