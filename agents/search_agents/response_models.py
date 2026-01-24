"""
Response Models - Pydantic models for validating LLM responses.

These models ensure that LLM responses conform to expected structures
and provide clear error messages when they don't.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from config import MAX_RECOMMENDED_VEHICLES


class VehicleRecommendation(BaseModel):
    """Single vehicle recommendation from the LLM."""
    
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model name")
    body_type: str = Field(..., description="Vehicle body type (SUV, Sedan, etc.)")
    years: str = Field(..., description="Years available")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Match score between 0 and 1")
    match_reason: str = Field(..., description="Explanation of why this vehicle matches user needs")
    
    @field_validator('match_score')
    @classmethod
    def validate_match_score(cls, v):
        """Ensure match score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"match_score must be between 0 and 1, got {v}")
        return v


class VehicleRecommendationResponse(BaseModel):
    """Complete response from vehicle model retriever."""
    
    vehicles: List[VehicleRecommendation] = Field(
        default_factory=list,
        description=f"List of recommended vehicles (up to {MAX_RECOMMENDED_VEHICLES})"
    )
    explanation: str = Field(..., description="Overall reasoning for the recommendations")
    
    @field_validator('vehicles')
    @classmethod
    def validate_vehicles_count(cls, v):
        """Ensure no more than MAX_RECOMMENDED_VEHICLES are returned."""
        if len(v) > MAX_RECOMMENDED_VEHICLES:
            raise ValueError(f"Maximum {MAX_RECOMMENDED_VEHICLES} vehicles allowed, got {len(v)}")
        return v


class ReasoningResponse(BaseModel):
    """Response for listing reasoning generation."""
    
    reasoning: str = Field(..., max_length=500, description="Brief reasoning (max 2 sentences)")
    
    @field_validator('reasoning')
    @classmethod
    def validate_reasoning_length(cls, v):
        """Ensure reasoning is concise."""
        sentences = v.split('.')
        if len([s for s in sentences if s.strip()]) > 2:
            raise ValueError("Reasoning should be maximum 2 sentences")
        return v
