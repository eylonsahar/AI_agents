"""
Listings Retriever - Class-based implementation for retrieving vehicle listings.

This module provides the ListingsRetriever class that:
1. Loads vehicle listings from the hardcoded CSV file
2. Filters listings based on vehicle models from Stage 1
3. Returns structured VehicleListing objects
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from config import LISTINGS_CSV_PATH, MAX_LISTINGS_PER_VEHICLE
from agents.utils.contracts import VehicleListing, VehicleModel, VehicleModelsResult


class ListingsRetriever:
    """
    Retrieves vehicle listings from CSV and filters them based on vehicle models.
    
    The CSV path is hardcoded in the configuration (LISTINGS_CSV_PATH).
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the ListingsRetriever.
        
        Args:
            csv_path: Optional path to CSV file. If not provided, uses LISTINGS_CSV_PATH from config.
        """
        self.csv_path = csv_path or LISTINGS_CSV_PATH
        self.df: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess the CSV data."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Listings CSV not found at: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Preprocess data
        if "id" in self.df.columns:
            self.df["id"] = self.df["id"].astype(str)
        
        # Normalize string columns
        for col in ("manufacturer", "model", "state", "region", "condition", "paint_color"):
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str).str.lower().str.strip()
        
        # Convert numeric columns
        for col in ("price", "year"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        
        print(f"Loaded {len(self.df)} listings from {self.csv_path}")
    
    def _count_info(self, car: Dict[str, Any]) -> int:
        """
        Count how many relevant fields are non-empty in a car listing.
        
        Args:
            car: Dictionary representing a car listing
            
        Returns:
            Count of non-empty fields
        """
        fields_to_check = [
            "region", "price", "year", "manufacturer", "model",
            "condition", "paint_color", "state", "posting_date"
        ]
        count = 0
        for f in fields_to_check:
            if car.get(f) not in (None, "", []):
                count += 1
        return count
    
    def _select_top_cars(self, cars: List[Dict[str, Any]], n: int = MAX_LISTINGS_PER_VEHICLE) -> List[Dict[str, Any]]:
        """
        Select up to n cars from the list based on:
        1. If length <= n, return all.
        2. Otherwise, pick cars with the newest posting_date.
        3. Then prioritize cars with more complete information.
        
        Args:
            cars: List of car dictionaries
            n: Number of cars to select
            
        Returns:
            List of top n cars
        """
        if len(cars) <= n:
            return cars
        
        # Parse posting_date to datetime objects
        for car in cars:
            try:
                car["_posting_dt"] = datetime.strptime(car["posting_date"][:19], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                car["_posting_dt"] = datetime.min  # fallback if date invalid or missing
        
        # Sort by posting_date descending (newest first)
        cars_sorted = sorted(cars, key=lambda x: x["_posting_dt"], reverse=True)
        # Then sort by completeness descending (more info first)
        cars_sorted = sorted(cars_sorted, key=lambda x: self._count_info(x), reverse=True)
        
        # Select top n
        top_n = cars_sorted[:n]
        
        # Clean up temporary field
        for car in top_n:
            del car["_posting_dt"]
        
        return top_n
    
    def _parse_years_range(self, years_value: Any) -> Optional[tuple[int, int]]:
        """
        Parse a years range string into min and max years.
        
        Args:
            years_value: String like "2018-2023" or "2020"
            
        Returns:
            Tuple of (min_year, max_year) or None
        """
        if years_value is None:
            return None
        
        text = str(years_value).strip()
        if not text:
            return None
        
        text = text.replace("–", "-").replace("—", "-")
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        if not matches:
            return None
        
        years = [int(y) for y in matches]
        if len(years) == 1:
            return years[0], years[0]
        
        return min(years), max(years)
    
    def retrieve_listings(
        self, 
        vehicle_models_result: VehicleModelsResult,
        top_n: int = MAX_LISTINGS_PER_VEHICLE
    ) -> List[VehicleListing]:
        """
        Retrieve listings for the given vehicle models.
        
        Args:
            vehicle_models_result: Result from Stage 1 containing recommended vehicle models
            top_n: Maximum number of listings to return per vehicle model
            
        Returns:
            List of VehicleListing objects
        """
        all_listings: List[VehicleListing] = []
        
        for vehicle_model in vehicle_models_result.vehicles:
            make = vehicle_model.make.strip().lower()
            model = vehicle_model.model.strip().lower()
            years_range = self._parse_years_range(vehicle_model.years)
            
            if not make and not model:
                continue
            
            # Build filter mask
            mask = pd.Series([True] * len(self.df))
            
            if make and "manufacturer" in self.df.columns:
                mask &= self.df["manufacturer"] == make
            
            if model and "model" in self.df.columns:
                mask &= self.df["model"].str.contains(model, case=False, na=False)
            
            if years_range is not None and "year" in self.df.columns:
                min_year, max_year = years_range
                mask &= self.df["year"].between(min_year, max_year, inclusive="both")
            
            # Get matching listings
            listings_raw = self.df[mask].to_dict(orient="records")
            
            # Select top n using custom ranking
            top_listings = self._select_top_cars(listings_raw, n=top_n)
            
            # Convert to VehicleListing objects
            for listing_dict in top_listings:
                listing = VehicleListing.from_dict(listing_dict)
                all_listings.append(listing)
        
        return all_listings
    
    def retrieve_listings_with_grouping(
        self, 
        vehicle_models_result: VehicleModelsResult,
        top_n: int = MAX_LISTINGS_PER_VEHICLE
    ) -> Dict[str, Any]:
        """
        Retrieve listings grouped by vehicle model (legacy format for compatibility).
        
        Args:
            vehicle_models_result: Result from Stage 1 containing recommended vehicle models
            top_n: Maximum number of listings to return per vehicle model
            
        Returns:
            Dictionary with format:
            {
                "results": [
                    {
                        "vehicle": {...vehicle model dict...},
                        "listings": [...list of listing dicts...]
                    },
                    ...
                ]
            }
        """
        results = []
        
        for vehicle_model in vehicle_models_result.vehicles:
            make = vehicle_model.make.strip().lower()
            model = vehicle_model.model.strip().lower()
            years_range = self._parse_years_range(vehicle_model.years)
            
            if not make and not model:
                listings = []
            else:
                # Build filter mask
                mask = pd.Series([True] * len(self.df))
                
                if make and "manufacturer" in self.df.columns:
                    mask &= self.df["manufacturer"] == make
                
                if model and "model" in self.df.columns:
                    mask &= self.df["model"].str.contains(model, case=False, na=False)
                
                if years_range is not None and "year" in self.df.columns:
                    min_year, max_year = years_range
                    mask &= self.df["year"].between(min_year, max_year, inclusive="both")
                
                # Get matching listings
                listings_raw = self.df[mask].to_dict(orient="records")
                
                # Select top n using custom ranking
                listings = self._select_top_cars(listings_raw, n=top_n)
            
            results.append({
                "vehicle": vehicle_model.to_dict(),
                "listings": listings
            })
        
        return {"results": results}


def retrieve_listings_from_csv(
    vehicles_result: Dict[str, Any], 
    listings_csv_path: Optional[str] = None, 
    top_n: int = MAX_LISTINGS_PER_VEHICLE
) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Retrieves listings from CSV file.
    """
    from agents.utils.contracts import VehicleModelsResult
    
    # Convert to VehicleModelsResult
    vmr = VehicleModelsResult.from_raw_result(
        query=vehicles_result.get('query', ''),
        raw_result=vehicles_result
    )
    
    # Create retriever
    retriever = ListingsRetriever(csv_path=listings_csv_path)
    
    return retriever.retrieve_listings_with_grouping(vmr, top_n)


if __name__ == "__main__":
    """Test the ListingsRetriever class."""
    import json
    
    # Create a test vehicle models result
    test_vehicles = {
        'query': 'reliable SUV for family',
        'vehicles': [
            {
                'make': 'Honda',
                'model': 'CR-V',
                'body_type': 'SUV',
                'years': '2018-2023',
                'match_score': 0.96,
                'match_reason': 'Great family car with safety features'
            }
        ]
    }
    
    from agents.utils.contracts import VehicleModelsResult
    vmr = VehicleModelsResult.from_raw_result(
        query=test_vehicles['query'],
        raw_result=test_vehicles
    )
    
    # Test the retriever
    retriever = ListingsRetriever()
    listings = retriever.retrieve_listings(vmr, top_n=3)
    
    print(f"Found {len(listings)} listings")
    for i, listing in enumerate(listings[:3], 1):
        print(f"\nListing {i}:")
        print(f"  ID: {listing.id}")
        print(f"  Make/Model: {listing.manufacturer} {listing.model}")
        print(f"  Year: {listing.year}")
        print(f"  Price: ${listing.price}")
        print(f"  State: {listing.state}")
