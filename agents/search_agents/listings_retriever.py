import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd


def retrieve_listings_from_csv(
    vehicles_result: Dict[str, Any],
    listings_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get JSON, return JSON.

    Input:
      vehicles_result: dict in the format of test_result.json:
        {
          "vehicles": [
            {
              "make": "...",
              "model": "...",
              ...
            },
            ...
          ]
        }

    Output:
      {
        "results": [
          {
            "vehicle": { ...original vehicle object... },
            "listings": [ { ...csv row... }, ... ]
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

            listings = df[mask].head(2).to_dict(orient="records")

        results.append(
            {
                "vehicle": vehicle,
                "listings": listings,
            }
        )

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