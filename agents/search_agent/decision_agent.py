from typing import Dict, Any, List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.utils.contracts import VehicleListing, VehicleModel, VehicleModelsResult, ScoredListing
from config import SUSPICIOUS_PRICE_THRESHOLD

SUSPICIOUS_PRICE_PENALTY = 0.3  # Multiplier applied to listing_score for suspicious-price cars



def min_max_norm(value: float, min_v: float, max_v: float) -> float:
    """Normalize a value between 0 and 1 using min-max normalization."""
    if max_v == min_v:
        return 0.5
    return (value - min_v) / (max_v - min_v)



def score_vehicle_model(vehicle_model: VehicleModel) -> Tuple[float, List[str]]:
    """
    Score a vehicle model based on its match score.
    
    Args:
        vehicle_model: VehicleModel object
        
    Returns:
        Tuple of (score, reasons)
    """
    match_score = vehicle_model.match_score
    reasons: List[str] = []

    if match_score >= 0.85:
        reasons.append("Strong overall match to buyer needs")
    elif match_score >= 0.65:
        reasons.append("Reasonable match with some trade-offs")
    else:
        reasons.append("Lower overall suitability compared to alternatives")

    return match_score, reasons


def score_listings(listings: List[VehicleListing]) -> Dict[int, Tuple[float, List[str]]]:
    """
    Score a list of vehicle listings based on multiple criteria:
    - Price (lower is better)
    - Year (newer is better)
    - Condition (excellent > good > fair)
    - Mileage (lower is better)
    - Accident history (none > minor > reported)

    Args:
        listings: List of VehicleListing objects
        
    Returns:
        Dictionary mapping index to (score, reasons)
    """
    # Collect valid values for normalization
    valid_prices = [l.price for l in listings if l.price and l.price > 0]
    valid_years = [l.year for l in listings if l.year]
    valid_mileages = []
    for l in listings:
        if l.mileage:
            try:
                mileage_val = float(str(l.mileage).replace(',', ''))
                if mileage_val > 0:
                    valid_mileages.append(mileage_val)
            except (ValueError, AttributeError):
                pass

    if not valid_prices or not valid_years:
        return {
            i: (0.0, ["Insufficient data for scoring"])
            for i in range(len(listings))
        }

    min_price, max_price = min(valid_prices), max(valid_prices)
    min_year, max_year = min(valid_years), max(valid_years)
    
    # Mileage normalization (if available)
    if valid_mileages:
        min_mileage, max_mileage = min(valid_mileages), max(valid_mileages)
    else:
        min_mileage, max_mileage = 0, 1

    results: Dict[int, Tuple[float, List[str]]] = {}

    for idx, listing in enumerate(listings):
        reasons: List[str] = []

        # Price scoring
        price = listing.price or 0
        if price <= 0:
            results[idx] = (0.0, ["Invalid or missing price"])
            continue
        suspicious_price = price < SUSPICIOUS_PRICE_THRESHOLD
        price_score = 1 - min_max_norm(price, min_price, max_price)

        # Year scoring
        year = listing.year or 0
        year_score = min_max_norm(year, min_year, max_year)

        # Condition scoring
        condition = (listing.condition or "").lower().strip()
        condition_map = {
            "excellent": 1.0,
            "like new": 0.95,
            "good": 0.7,
            "fair": 0.4,
            "salvage": 0.1,
        }
        condition_score = condition_map.get(condition, 0.3)

        # Mileage scoring
        mileage_score = 0.5  # Default if not available
        if listing.mileage:
            try:
                mileage_val = float(str(listing.mileage).replace(',', ''))
                if mileage_val > 0 and valid_mileages:
                    # Lower mileage is better, so invert the normalization
                    mileage_score = 1 - min_max_norm(mileage_val, min_mileage, max_mileage)
            except (ValueError, AttributeError):
                pass

        # Accident history scoring
        accident_score = 0.7  # Default if not available
        if listing.accident is not None:
            accident_score = 1.0 if not listing.accident else 0.1

        # Weighted listing score
        # Price: 30%, Year: 25%, Condition: 20%, Mileage: 15%, Accident: 10%
        listing_score = (
            0.30 * price_score +
            0.25 * year_score +
            0.20 * condition_score +
            0.15 * mileage_score +
            0.10 * accident_score
        )

        # Suspicious price penalty — applied before generating reasons
        if suspicious_price:
            listing_score *= SUSPICIOUS_PRICE_PENALTY
            reasons.append(f"⚠️ Suspicious price (${price:.0f}) — could not be verified with seller")

        # Generate reasons
        if price_score >= 0.7 and not suspicious_price:
            reasons.append("Competitive price")
        if year_score >= 0.7:
            reasons.append("Newer model year")
        if condition_score >= 0.7:
            reasons.append("Excellent condition")
        if mileage_score >= 0.7:
            reasons.append("Low mileage")
        if accident_score >= 0.9:
            reasons.append("Clean accident history")
        
        # Negative reasons
        if accident_score < 0.5:
            reasons.append("Multiple accidents reported")
        if mileage_score < 0.3:
            reasons.append("High mileage")

        results[idx] = (listing_score, reasons)

    return results


class DecisionAgent:
    """
    Agent responsible for ranking and scoring vehicle listings.
    """
    
    def __init__(self):
        """Initialize the DecisionAgent."""
        pass

    
    def get_scored_listings(
        self, 
        listings: List[VehicleListing],
        vehicle_models_result: VehicleModelsResult
    ) -> List[ScoredListing]:
        """
        Get listings with full scoring information.
        
        Args:
            listings: List of VehicleListing objects to rank
            vehicle_models_result: VehicleModelsResult containing the recommended models
            
        Returns:
            List of ScoredListing objects ordered by final score (highest first)
        """
        # Create a mapping of (make, model) -> VehicleModel for quick lookup
        vehicle_model_map = {}
        for vm in vehicle_models_result.vehicles:
            key = (vm.make.lower(), vm.model.lower())
            vehicle_model_map[key] = vm
        
        scored_listings: List[ScoredListing] = []
        
        # Group listings by vehicle model for comparative scoring
        listings_by_model: Dict[str, List[Tuple[int, VehicleListing]]] = {}
        for idx, listing in enumerate(listings):
            if listing.manufacturer and listing.model:
                key = f"{listing.manufacturer}_{listing.model}"
                if key not in listings_by_model:
                    listings_by_model[key] = []
                listings_by_model[key].append((idx, listing))
        
        # Score each group
        for model_key, model_listings in listings_by_model.items():
            # Get the VehicleModel for this group
            if model_listings:
                first_listing = model_listings[0][1]
                vm_key = (first_listing.manufacturer.lower(), first_listing.model.lower())
                vehicle_model = vehicle_model_map.get(vm_key)
                
                if not vehicle_model:
                    continue
                
                # Get model score
                model_score, model_reasons = score_vehicle_model(vehicle_model)
                
                # Get listing scores for this group
                group_listings = [l for _, l in model_listings]
                listing_scores = score_listings(group_listings)
                
                # Create scored listings
                for group_idx, (original_idx, listing) in enumerate(model_listings):
                    listing_score, listing_reasons = listing_scores.get(group_idx, (0.0, []))
                    
                    # Final score: 45% model match + 55% listing quality
                    final_score = (
                        0.45 * model_score +
                        0.55 * listing_score
                    )
                    
                    scored_listings.append(
                        ScoredListing(
                            listing=listing,
                            vehicle_model=vehicle_model,
                            model_score=model_score,
                            listing_score=listing_score,
                            final_score=final_score,
                            reasons=model_reasons + listing_reasons
                        )
                    )
        
        # Sort by final score (highest first)
        scored_listings.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_listings
