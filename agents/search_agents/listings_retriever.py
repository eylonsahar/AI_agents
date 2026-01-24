import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
from config import MAX_LISTINGS_PER_VEHICLE


def count_info(car: Dict[str, Any]) -> int:
    """
    Count how many relevant fields are non-empty in a car listing.
    """
    fields_to_check = [
        "region",
        "price",
        "year",
        "manufacturer",
        "model",
        "condition",
        "paint_color",
        "state",
        "posting_date",
    ]
    count = 0
    for f in fields_to_check:
        if car.get(f) not in (None, "", []):
            count += 1
    return count


def select_top_cars(cars: List[Dict[str, Any]], n: int = MAX_LISTINGS_PER_VEHICLE) -> List[Dict[str, Any]]:
    """
    Select up to n cars from the list based on:
    1. If length <= n, return all.
    2. Otherwise, pick cars with the newest posting_date.
    3. Then prioritize cars with more complete information.
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
    cars_sorted = sorted(cars_sorted, key=lambda x: count_info(x), reverse=True)

    # Select top n
    top_n = cars_sorted[:n]

    # Clean up temporary field
    for car in top_n:
        del car["_posting_dt"]

    return top_n


def retrieve_listings_from_csv(
    vehicles_result: Dict[str, Any], listings_csv_path: Optional[str] = None, top_n: int = MAX_LISTINGS_PER_VEHICLE
) -> Dict[str, Any]:
    """
    Given a dict of vehicles, retrieve matching listings from CSV, filter, and
    return results with top n listings per vehicle.

    Input:
      vehicles_result: dict in format like test_result.json, contains "vehicles" list.
      listings_csv_path: optional path to CSV file with vehicle listings.
      top_n: how many listings to return per vehicle.

    Output:
      {
        "results": [
          {
            "vehicle": {...original vehicle object...},
            "listings": [ ...top n filtered listings... ]
          },
          ...
        ]
      }
    """
    if listings_csv_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        listings_csv_path = os.path.join(project_root, "rag", "data", "cars_for_sale.csv")

    df = pd.read_csv(listings_csv_path)

    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    for col in ("manufacturer", "model", "state", "region", "condition", "paint_color"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.lower().str.strip()

    for col in ("price", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    vehicles: List[Dict[str, Any]] = vehicles_result.get("vehicles", []) or []
    results: List[Dict[str, Any]] = []

    for vehicle in vehicles:
        make = str(vehicle.get("make", "")).strip().lower()
        model = str(vehicle.get("model", "")).strip().lower()

        if not make and not model:
            listings = []
        else:
            mask = pd.Series([True] * len(df))

            if make and "manufacturer" in df.columns:
                mask &= df["manufacturer"] == make

            if model and "model" in df.columns:
                mask &= df["model"].str.contains(model, case=False, na=False)

            listings_raw = df[mask].to_dict(orient="records")
            # Select top n using custom ranking function
            listings = select_top_cars(listings_raw, n=top_n)

        results.append({"vehicle": vehicle, "listings": listings})

    return {"results": results}


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    test_json_path = os.path.join(base_dir, "test_result2_suv.json")

    if not os.path.exists(test_json_path):
        raise SystemExit(f"test_result.json not found at {test_json_path}")

    with open(test_json_path, "r", encoding="utf-8") as f:
        vehicles_result = json.load(f)

    output = retrieve_listings_from_csv(vehicles_result)
    print(json.dumps(output, indent=2, ensure_ascii=False))
