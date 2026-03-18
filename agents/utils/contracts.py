from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class VehicleModel:
    make: str
    model: str
    body_type: str
    years: str
    match_score: float
    match_reason: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleModel':
        """Create VehicleModel from dictionary."""
        return cls(
            make=data.get('make', ''),
            model=data.get('model', ''),
            body_type=data.get('body_type', ''),
            years=data.get('years', ''),
            match_score=data.get('match_score', 0.0),
            match_reason=data.get('match_reason', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'make': self.make,
            'model': self.model,
            'body_type': self.body_type,
            'years': self.years,
            'match_score': self.match_score,
            'match_reason': self.match_reason
        }


@dataclass
class VehicleModelsResult:
    query: str
    vehicles: List[VehicleModel]
    explanation: str
    raw_result: Dict[str, Any]  # Complete result from vehicle_retriever
    
    @classmethod
    def from_raw_result(cls, query: str, raw_result: Dict[str, Any]) -> 'VehicleModelsResult':
        """Create VehicleModelsResult from raw retriever output."""
        vehicles_data = raw_result.get('vehicles', [])
        vehicles = [VehicleModel.from_dict(v) for v in vehicles_data]
        explanation = raw_result.get('explanation', '')
        return cls(
            query=query,
            vehicles=vehicles,
            explanation=explanation,
            raw_result=raw_result
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return self.raw_result


@dataclass
class ScoredListing:
    listing: 'VehicleListing'
    vehicle_model: VehicleModel
    final_score: float
    model_score: float
    listing_score: float
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'listing': self.listing.to_dict(),
            'vehicle_model': self.vehicle_model.to_dict(),
            'final_score': self.final_score,
            'model_score': self.model_score,
            'listing_score': self.listing_score,
            'reasons': self.reasons
        }


@dataclass
class VehicleListing:
    id: str
    region: Optional[str] = None
    price: Optional[float] = None
    year: Optional[int] = None
    condition: Optional[str] = None
    paint_color: Optional[str] = None
    state: Optional[str] = None
    posting_date: Optional[str] = None
    mileage: Optional[str] = None
    accident: Optional[bool] = None
    meetings: Optional[List[Any]] = None
    
    # Additional fields from CSV that may be present
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    list_price: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleListing':
        """Create VehicleListing from dictionary."""
        return cls(
            id=str(data.get('id', '')),
            region=data.get('region'),
            price=data.get('price'),
            year=int(v) if (v := data.get('year')) is not None and str(v) not in ('', '<NA>') else None,
            condition=data.get('condition'),
            paint_color=data.get('paint_color'),
            state=data.get('state'),
            posting_date=data.get('posting_date'),
            mileage=data.get('mileage'),
            accident=data.get('accident'),
            meetings=data.get('meetings', []),
            manufacturer=data.get('manufacturer'),
            model=data.get('model'),
            list_price=float(v) if (v := data.get('list_price')) is not None and str(v) not in ('', '<NA>') else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'id': self.id,
            'region': self.region,
            'price': self.price,
            'year': self.year,
            'condition': self.condition,
            'paint_color': self.paint_color,
            'state': self.state,
            'posting_date': self.posting_date,
            'mileage': self.mileage,
            'accident': self.accident,
            'meetings': self.meetings
        }
        if self.manufacturer:
            result['manufacturer'] = self.manufacturer
        if self.model:
            result['model'] = self.model
        if self.list_price is not None:
            result['list_price'] = self.list_price
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class RecommendationOutput:
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
    
    - budget_min, budget_max
    - preferred_location
    - preferred_condition
    - preferred_years
    """
    query_text: str  # Natural language query
    top_n_models: int = 5  # Number of models to retrieve in Stage 1
    top_k_listings: int = 10  # Number of listings to return in Stage 2
    max_price: Optional[float] = None
    min_year: Optional[int] = None
    preferred_state: Optional[str] = None


@dataclass
class PipelineResult:
    """
    Result from the complete search pipeline.
    
    Contains the essential results:
    - query: The original search query
    - vehicle_models_result: Stage 1 result with vehicle models
    - scored_listings: Stage 3 detailed scoring results
    - action_log: Log of all LLM calls made during the pipeline
    """
    query: str
    vehicle_models_result: Optional[VehicleModelsResult] = None
    scored_listings: Optional[List[ScoredListing]] = None
    action_log: Optional[Any] = None  # ActionLog instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "query": self.query,
            "vehicle_models_result": self.vehicle_models_result.to_dict() if self.vehicle_models_result else None,
            "scored_listings": [sl.to_dict() for sl in (self.scored_listings or [])]
        }
        return result
    
    def print_action_log(self) -> None:
        """Print the action log if available."""
        if self.action_log:
            self.action_log.print_steps()

