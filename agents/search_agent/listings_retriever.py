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
        
        # Convert numeric columns — use nullable Int64 so NaN rows don't
        # force the whole column to float (which would serialize as 2016.0 in JSON)
        for col in ("price", "year"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")
        
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
        
        # Sort by posting_date descending (newest first) and completeness descending (more info first)
        # Combined sort key: (posting_date, completeness) - tuple ensures both criteria are considered
        cars_sorted = sorted(
            cars, 
            key=lambda x: (x["_posting_dt"], self._count_info(x)), 
            reverse=True
        )
        
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
        top_n: int = MAX_LISTINGS_PER_VEHICLE,
        year_min: Optional[int] = None,
        price_max: Optional[float] = None,
    ) -> List[VehicleListing]:
        """
        Retrieve listings for the given vehicle models.

        Uses a tiered fallback strategy per model:
          1. manufacturer + model + model's production-year range
          2. manufacturer + model + user's year_min (drops model year range)
          3. manufacturer only + user's year_min (drops specific model name)

        Args:
            vehicle_models_result: Result from Stage 1 containing recommended vehicle models
            top_n: Maximum number of listings to return per vehicle model
            year_min: User's minimum year constraint (used in fallback tiers)
            price_max: User's maximum price constraint applied globally

        Returns:
            List of VehicleListing objects
        """
        all_listings: List[VehicleListing] = []

        for vehicle_model in vehicle_models_result.vehicles:
            make  = vehicle_model.make.strip().lower()
            model = vehicle_model.model.strip().lower()
            model_years = self._parse_years_range(vehicle_model.years)

            if not make and not model:
                continue

            # -----------------------------------------------------------
            # Tier 1: manufacturer + model + model production-year range
            # -----------------------------------------------------------
            listings_raw = self._query(make=make, model=model,
                                       year_range=model_years,
                                       year_min=year_min, price_max=price_max)

            # -----------------------------------------------------------
            # Tier 2: manufacturer + model + user year_min only
            # -----------------------------------------------------------
            if not listings_raw and model and year_min is not None:
                print(f"[ListingsRetriever] Tier-1 returned 0 for '{make} {model}'. "
                      f"Falling back to user year_min={year_min}.")
                listings_raw = self._query(make=make, model=model,
                                           year_range=None,
                                           year_min=year_min, price_max=price_max)

            # -----------------------------------------------------------
            # Tier 3: manufacturer only + user year_min
            # -----------------------------------------------------------
            if not listings_raw and make and year_min is not None:
                print(f"[ListingsRetriever] Tier-2 returned 0 for '{make} {model}'. "
                      f"Falling back to manufacturer '{make}' only.")
                listings_raw = self._query(make=make, model=None,
                                           year_range=None,
                                           year_min=year_min, price_max=price_max)

            top_listings = self._select_top_cars(listings_raw, n=top_n)

            for listing_dict in top_listings:
                listing = VehicleListing.from_dict(listing_dict)
                all_listings.append(listing)

        return all_listings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(
        self,
        make: Optional[str],
        model: Optional[str],
        year_range: Optional[tuple],
        year_min: Optional[int],
        price_max: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Apply filters to self.df and return matching rows as dicts."""
        mask = pd.Series([True] * len(self.df))

        if make and "manufacturer" in self.df.columns:
            mask &= self.df["manufacturer"] == make

        if model and "model" in self.df.columns:
            mask &= self.df["model"].str.contains(model, case=False, na=False)

        if "year" in self.df.columns:
            if year_range is not None:
                min_y, max_y = year_range
                # Clamp lower bound to user's year_min (fixes Bug #6/#9)
                if year_min is not None:
                    min_y = max(min_y, year_min)
                mask &= self.df["year"].between(min_y, max_y, inclusive="both")
            elif year_min is not None:
                mask &= self.df["year"] >= year_min

        if price_max is not None and "price" in self.df.columns:
            mask &= self.df["price"] <= price_max

        return self.df[mask].to_dict(orient="records")
