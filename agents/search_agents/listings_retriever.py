"""
Listings Retriever - Stage 2 Structured Retrieval.

This module performs structured (non-vector) retrieval on vehicle listings
from cars_for_sale.csv. It filters and scores listings based on the vehicle
models retrieved in Stage 1.

NOTE: This stage does NOT use embeddings or vector search.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from typing import List, Dict, Any, Optional
from Rag.agents.contracts import VehicleModel, VehicleListing
from Rag.agents.config_vehicles import (
    TOP_K_LISTINGS,
    LISTING_SCORING_WEIGHTS
)


class ListingsRetriever:
    """
    Structured retrieval for vehicle listings.
    
    Filters and scores listings from cars_for_sale.csv based on:
    - Vehicle models from Stage 1
    - User preferences (price, year, condition, location)
    """
    
    def __init__(self, listings_csv_path: str):
        """
        Initialize the listings retriever.
        
        Args:
            listings_csv_path: Path to cars_for_sale.csv
        """
        self.listings_csv_path = listings_csv_path
        self.listings_df = None
        self._load_listings()
    
    def _load_listings(self):
        """Load listings from CSV file."""
        print(f"Loading listings from {self.listings_csv_path}...")
        self.listings_df = pd.read_csv(self.listings_csv_path)
        
        # Clean and preprocess data
        self.listings_df['id'] = self.listings_df['id'].astype(str)
        self.listings_df['manufacturer'] = self.listings_df['manufacturer'].fillna('').str.lower().str.strip()
        self.listings_df['model'] = self.listings_df['model'].fillna('').str.lower().str.strip()
        self.listings_df['price'] = pd.to_numeric(self.listings_df['price'], errors='coerce')
        self.listings_df['year'] = pd.to_numeric(self.listings_df['year'], errors='coerce')
        
        # Drop rows with missing critical fields
        self.listings_df = self.listings_df.dropna(subset=['manufacturer', 'model', 'price', 'year'])
        
        print(f"Loaded {len(self.listings_df)} valid listings")
    
    def _compute_listing_score(
        #TODO: this is not need to by score- it need to by by llm with rag of gaidline
        self,
        listing: pd.Series,
        vehicle_model: VehicleModel,
        max_price: Optional[float] = None,
        min_year: Optional[int] = None
    ) -> float:
        """
        Compute a score for a listing based on multiple factors.
        
        Args:
            listing: Pandas Series representing a listing
            vehicle_model: VehicleModel from Stage 1
            max_price: Optional maximum price filter
            min_year: Optional minimum year filter
        
        Returns:
            Normalized score between 0 and 1
        """
        score = 0.0
        weights = LISTING_SCORING_WEIGHTS
        
        # 1. Model match score (exact match with Stage 1 model)
        model_match = 0.0
        listing_make = listing['manufacturer'].lower()
        listing_model = listing['model'].lower()
        stage1_make = vehicle_model.make.lower()
        stage1_model = vehicle_model.model.lower()
        
        if listing_make == stage1_make and listing_model == stage1_model:
            model_match = 1.0
        elif listing_model == stage1_model:
            model_match = 0.7  # Model matches but not make
        elif stage1_model in listing_model or listing_model in stage1_model:
            model_match = 0.5  # Partial model match
        
        score += weights['model_match'] * model_match
        
        # 2. Price score (lower is better, normalized)
        # TODO: User should define price normalization strategy
        price_score = 0.0
        if pd.notna(listing['price']) and listing['price'] > 0:
            if max_price:
                # Normalize: price closer to 0 = higher score, price at max_price = 0 score
                price_score = max(0, 1 - (listing['price'] / max_price))
            else:
                # Default: assume $50k as reference max
                price_score = max(0, 1 - (listing['price'] / 50000))
        
        score += weights['price'] * price_score
        
        # 3. Year score (newer is better)
        year_score = 0.0
        if pd.notna(listing['year']) and listing['year'] > 1900:
            current_year = 2026  # TODO: Use dynamic current year
            age = current_year - listing['year']
            # Normalize: 0 years old = 1.0, 30+ years old = 0.0
            year_score = max(0, 1 - (age / 30))
        
        score += weights['year'] * year_score
        
        # 4. Condition score
        condition_score = 0.0
        condition_map = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.5,
            'salvage': 0.2,
            'new': 1.0,
            'like new': 0.9
        }
        condition = str(listing.get('condition', '')).lower()
        condition_score = condition_map.get(condition, 0.3)  # Default to 0.3 if unknown
        
        score += weights['condition'] * condition_score
        
        return score
    
    def retrieve_listings(
        self,
        vehicle_models: List[VehicleModel],
        top_k: int = TOP_K_LISTINGS,
        max_price: Optional[float] = None,
        min_year: Optional[int] = None,
        preferred_state: Optional[str] = None
    ) -> List[VehicleListing]:
        """
        Retrieve and score listings for the given vehicle models.
        
        Args:
            vehicle_models: List of VehicleModel objects from Stage 1
            top_k: Number of listings to return
            max_price: Optional maximum price filter
            min_year: Optional minimum year filter
            preferred_state: Optional state filter
        
        Returns:
            List of VehicleListing objects ranked by score
        """
        all_listings = []
        
        for vehicle_model in vehicle_models:
            # Filter listings by manufacturer and model
            make_lower = vehicle_model.make.lower()
            model_lower = vehicle_model.model.lower()
            
            filtered = self.listings_df[
                (self.listings_df['manufacturer'] == make_lower) &
                (self.listings_df['model'].str.contains(model_lower, case=False, na=False))
            ]
            
            # Apply optional filters
            if max_price:
                filtered = filtered[filtered['price'] <= max_price]
            
            if min_year:
                filtered = filtered[filtered['year'] >= min_year]
            
            if preferred_state:
                filtered = filtered[filtered['state'] == preferred_state.lower()]
            
            # Score each listing
            for _, listing in filtered.iterrows():
                score = self._compute_listing_score(
                    listing,
                    vehicle_model,
                    max_price,
                    min_year
                )
                
                vehicle_listing = VehicleListing(
                    listing_id=str(listing['id']),
                    vehicle_model=f"{vehicle_model.make} {vehicle_model.model}",
                    manufacturer=listing['manufacturer'],
                    model=listing['model'],
                    price=float(listing['price']),
                    year=int(listing['year']),
                    condition=listing.get('condition'),
                    paint_color=listing.get('paint_color'),
                    state=listing.get('state'),
                    region=listing.get('region'),
                    score=score
                )
                all_listings.append(vehicle_listing)
        
        # Sort by score (descending) and return top K
        all_listings.sort(key=lambda x: x.score, reverse=True)
        return all_listings[:top_k]
    
    def get_listings_summary(
        self,
        vehicle_models: List[VehicleModel],
        top_k: int = TOP_K_LISTINGS,
        **filters
    ) -> Dict[str, Any]:
        """
        Get a summary of retrieved listings.
        
        Args:
            vehicle_models: Vehicle models from Stage 1
            top_k: Number of listings to retrieve
            **filters: Optional filters (max_price, min_year, preferred_state)
        
        Returns:
            Dictionary with listings and statistics
        """
        listings = self.retrieve_listings(vehicle_models, top_k, **filters)
        
        return {
            'num_models_searched': len(vehicle_models),
            'num_listings_found': len(listings),
            'top_k': top_k,
            'filters': filters,
            'listings': [
                {
                    'listing_id': l.listing_id,
                    'vehicle_model': l.vehicle_model,
                    'price': l.price,
                    'year': l.year,
                    'condition': l.condition,
                    'state': l.state,
                    'score': l.score
                }
                for l in listings
            ]
        }
