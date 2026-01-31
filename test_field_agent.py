from agents.field_agent.field_agent import FieldAgent
from agents.action_log import ActionLog

CLEANED_PAYLOAD = {
    "results": [
        {
            "vehicle": {
                "make": "Kia",
                "model": "Sportage",
                "body_type": "Estate",
                "years": "2022-present",
                "match_score": 0.95,
                "match_reason": "Family SUV with strong dependability."
            },
            "listings": [
                {
                    "id": "7315366439",
                    "region": "birmingham",
                    "price": 11400,
                    "year": 2013.0,
                    "manufacturer": "kia",
                    "model": "sportage",
                    "condition": "",           # MISSING
                    "paint_color": "blue",
                    "state": "al",
                    "posting_date": "2021-05-01T12:01:16-0500"
                    # MISSING: mileage, accident
                },
                {
                    "id": "7304888907",
                    "region": "birmingham",
                    "price": 19900,
                    "year": 2018.0,
                    "manufacturer": "kia",
                    "model": "sportage",
                    "condition": "good",
                    "paint_color": "",         # MISSING
                    "state": "al",
                    "posting_date": "2021-04-10T15:57:05-0500"
                    # MISSING: mileage, accident
                }
            ]
        },
        {
            "vehicle": {
                "make": "Honda",
                "model": "Cr V",
                "body_type": "Estate",
                "years": "2023-present",
                "match_score": 0.85,
                "match_reason": "Upmarket family SUV with high reliability."
            },
            "listings": [
                {
                    "id": "7313858171",
                    "region": "birmingham",
                    "price": 0,                # BAD - price is 0
                    "year": 2018.0,
                    "manufacturer": "honda",
                    "model": "cr v",
                    "condition": "excellent",
                    "paint_color": "red",
                    "state": "",               # MISSING
                    "posting_date": "2021-04-28T11:01:45-0500"
                    # MISSING: mileage, accident
                },
                {
                    "id": "7303674215",
                    "region": "gadsden-anniston",
                    "price": 0,                # BAD - price is 0
                    "year": 2019.0,
                    "manufacturer": "honda",
                    "model": "cr v",
                    "condition": "excellent",
                    "paint_color": "red",
                    "state": "al",
                    "posting_date": "2021-04-08T11:01:13-0500"
                    # MISSING: mileage, accident
                }
            ]
        },
        {
            "vehicle": {
                "make": "Hyundai",
                "model": "Tucson",
                "body_type": "4X4",
                "years": "2015-2020",
                "match_score": 0.65,
                "match_reason": "Practical SUV option with sensible features."
            },
            "listings": [
                {
                    "id": "7315461004",
                    "region": "dothan",
                    "price": "",               # MISSING
                    "year": 2013.0,
                    "manufacturer": "hyundai",
                    "model": "tucson",
                    "condition": "excellent",
                    "paint_color": "white",
                    "state": "al",
                    "posting_date": "2021-05-01T14:40:10-0500"
                    # MISSING: mileage, accident
                },
                {
                    "id": "7313657444",
                    "region": "huntsville / decatur",
                    "price": 3900,
                    "year": "",                # MISSING
                    "manufacturer": "hyundai",
                    "model": "tucson",
                    "condition": "",           # MISSING
                    "paint_color": "white",
                    "state": "al",
                    "posting_date": "2021-04-27T20:39:04-0500"
                    # MISSING: mileage, accident
                }
            ]
        }
    ]
}


if __name__ == '__main__':
    # ActionLog is owned here and passed into the agent (same pattern the
    # Supervisor uses at runtime).
    action_log = ActionLog()

    agent = FieldAgent(ads_list=CLEANED_PAYLOAD, action_log=action_log)
    results = agent.process_listings()

    # ── print the shared log (every LLM call the agent made) ──
    action_log.print_steps()

    # ── sanity checks ──
    print("\n📊 Results stats:", results["stats"])

    steps = action_log.get_steps()
    print(f"\n📋 Total steps logged: {len(steps)}")

    # 1. Every step must have the four required schema keys
    required_keys = {"module", "submodule", "prompt", "response"}
    for i, step in enumerate(steps):
        missing = required_keys - step.keys()
        assert not missing, f"Step {i} missing keys: {missing}"
    print("✅ All steps have correct schema (module, submodule, prompt, response)")

    # 2. We expect all three FieldAgent submodules to appear at least once
    expected_submodules = {"DecisionMaking", "MockSeller/GetData", "MockSeller/Scheduling"}
    seen_submodules = {s["submodule"] for s in steps}
    assert expected_submodules <= seen_submodules, (
        f"Missing expected submodules: {expected_submodules - seen_submodules}"
    )
    print("✅ All expected submodules present:", sorted(seen_submodules))

    # 3. Every listing must have meetings scheduled
    for group in results["results"]:
        for listing in group["listings"]:
            assert listing.get("meetings"), f"Listing {listing['id']} has no meetings"
    print("✅ Every listing has meetings scheduled")
