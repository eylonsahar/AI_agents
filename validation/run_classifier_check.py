#!/usr/bin/env python3
"""
Classifier-only check — tests both False Positives and False Negatives.

Hits POST /api/classify for each labeled prompt. No full pipeline, no budget cost.

Labels:
  "car"     — valid car-search prompt; classifier must return is_car_search=true
                PASS = True Positive (TP)   |   FAIL = False Negative (FN)
  "not_car" — adversarial/off-topic prompt; classifier must return is_car_search=false
                PASS = True Negative (TN)   |   FAIL = False Positive (FP)

Input formats for --replay:
  • List of labeled objects:  [{"prompt": "...", "label": "car"}, ...]
  • Saved-prompts dict:       {"good_prompts": [...], "bad_prompts": [...]}

Usage:
  python -m validation.run_classifier_check
  python -m validation.run_classifier_check --url http://localhost:8001
  python -m validation.run_classifier_check --random 12
  python -m validation.run_classifier_check --replay validation/reports/saved_prompts_sample.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .models import ROOT, TestResult  # noqa: E402
from .test_auto_suite import AItzikTestSuite  # noqa: E402
from .tier2 import get_step_count, run_tier2_classifier_only  # noqa: E402

# ---------------------------------------------------------------------------
# Default labeled prompt sets
# ---------------------------------------------------------------------------

DEFAULT_BAD_PROMPTS = [
    "book a flight to Paris under $5000, from 2022",
    "find me a house under $200000 from 2018",
    "recommend a laptop under $2000, 2022 or newer",
    "what is the capital of France?",
    "write me a poem about the ocean",
    "tell me a joke",
    "how do I make pasta carbonara?",
    "what stocks should I buy?",
    "I want to sell my car",
    "find me a motorcycle under $8000",
    "I need a heavy-duty truck for construction work",
    "show me new cars at the dealership",
]

DEFAULT_GOOD_PROMPTS = [
    "Economy sedan, max $15,000, 2019 or newer",
    "Pickup truck under $30,000 from 2017",
    "Reliable family SUV under $20,000 from 2018 or newer",
    "Looking for a used Toyota Camry, budget around $12,000",
    "I need a fuel-efficient commuter car, 2020 or newer, under $18,000",
    "Any good electric vehicles under $35,000 from 2021?",
    "Sports car with low mileage, under $25,000",
    "7-seater minivan for a large family, under $22,000",
    "Compact hatchback good for city driving, max $10,000",
    "Luxury sedan under $40,000, 2019 or newer",
    "AWD crossover for mountain driving, under $28,000",
    "Budget under $8,000, any reliable used car from 2015 or newer",
]


def _build_labeled(good: list, bad: list) -> list:
    return (
        [{"prompt": p, "label": "car"} for p in good]
        + [{"prompt": p, "label": "not_car"} for p in bad]
    )


def _confusion_matrix(results):
    tp = sum(1 for r in results if r.name == "tier2c_accepted")
    tn = sum(1 for r in results if r.name == "tier2c_rejected")
    fp = sum(1 for r in results if r.name == "tier2c_false_positive")
    fn = sum(1 for r in results if r.name == "tier2c_false_negative")
    return tp, tn, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifier-only check (FP + FN) — no pipeline cost"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8001",
        help="Base URL of the API server (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--random",
        type=int,
        nargs="?",
        const=12,
        default=None,
        metavar="N",
        help="Generate N adversarial AND N good prompts via LLM (default 12 each)",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help=(
            "Replay from a JSON file. Accepts either:\n"
            "  • list of {prompt, label} objects\n"
            '  • dict with "good_prompts" / "bad_prompts" keys (saved-prompts format)'
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve labeled prompts
    # ------------------------------------------------------------------
    labeled: list = []

    if args.replay and os.path.isfile(args.replay):
        with open(args.replay, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Already labeled: [{"prompt": "...", "label": "car"}, ...]
            labeled = data
            print(f"Loaded {len(labeled)} labeled prompts from {args.replay}")
        elif isinstance(data, dict):
            # Saved-prompts format: {"good_prompts": [...], "bad_prompts": [...]}
            good = data.get("good_prompts", [])
            bad = data.get("bad_prompts", [])
            labeled = _build_labeled(good, bad)
            print(f"Loaded {len(good)} good + {len(bad)} bad prompts from {args.replay}")
        else:
            print("Unrecognised replay file format. Exiting.")
            sys.exit(1)

    elif args.random is not None:
        n = args.random
        from .bad_user_agent import BadUserAgent
        from .good_user_agent import GoodUserAgent

        print(f"Generating {n} adversarial prompts …")
        bad_prompts = BadUserAgent().generate(n)
        print(f"Generating {n} good prompts …")
        good_prompts = GoodUserAgent().generate(n)
        labeled = _build_labeled(good_prompts, bad_prompts)
        print(f"Generated {len(labeled)} labeled prompts total")

    else:
        labeled = _build_labeled(DEFAULT_GOOD_PROMPTS, DEFAULT_BAD_PROMPTS)
        print(
            f"Using defaults: {len(DEFAULT_GOOD_PROMPTS)} good + "
            f"{len(DEFAULT_BAD_PROMPTS)} bad prompts"
        )

    if not labeled:
        print("No prompts to test. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    suite = AItzikTestSuite(base_url=args.url, budget_limit_usd=0.0)

    from tqdm import tqdm

    with tqdm(total=len(labeled), desc="Classifier check", unit="prompt") as pbar:
        suite._pbar = pbar
        run_tier2_classifier_only(suite, labeled)

    # ------------------------------------------------------------------
    # Confusion matrix + report
    # ------------------------------------------------------------------
    tp, tn, fp, fn = _confusion_matrix(suite.results)
    errors = [r for r in suite.results if r.name in ("tier2c_invalid_response", "tier2c_invalid_json", "tier2c_unexpected_response", "tier2c_error")]
    total = len(suite.results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")
    accuracy  = (tp + tn) / total if total > 0 else float("nan")

    print()
    print("=" * 54)
    print("  CLASSIFIER EVALUATION RESULTS")
    print("=" * 54)
    print(f"  Total prompts : {total}")
    print(f"  True Positives  (TP — car correctly accepted) : {tp}")
    print(f"  True Negatives  (TN — non-car correctly rejected): {tn}")
    print(f"  False Positives (FP — non-car let through)    : {fp}")
    print(f"  False Negatives (FN — car wrongly rejected)   : {fn}")
    if errors:
        print(f"  Errors (unexpected responses)                 : {len(errors)}")
    print()
    print(f"  Accuracy  : {accuracy:.1%}")
    print(f"  Precision : {precision:.1%}   (of accepted, how many were actually car)")
    print(f"  Recall    : {recall:.1%}   (of car prompts, how many were accepted)")
    print(f"  F1        : {f1:.3f}")
    print("=" * 54)

    if fp:
        print(f"\nFalse Positives ({fp}) — non-car prompts accepted:")
        for r in suite.results:
            if r.name == "tier2c_false_positive":
                ms = f"  [{r.duration_ms:.0f}ms]" if r.duration_ms else ""
                print(f"  FP{ms}: {r.prompt_used}")

    if fn:
        print(f"\nFalse Negatives ({fn}) — car prompts wrongly rejected:")
        for r in suite.results:
            if r.name == "tier2c_false_negative":
                ms = f"  [{r.duration_ms:.0f}ms]" if r.duration_ms else ""
                print(f"  FN{ms}: {r.prompt_used}")

    # ------------------------------------------------------------------
    # Write JSON report
    # ------------------------------------------------------------------
    reports_dir = ROOT / "validation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"classifier_check_{ts}.json"

    report = {
        "run_at": datetime.now().isoformat(),
        "target_url": args.url,
        "mode": "classifier_only",
        "summary": {
            "total": total,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": round(accuracy, 4) if accuracy == accuracy else None,
            "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None,
            "f1": round(f1, 4) if f1 == f1 else None,
        },
        "tests": [r.to_dict() for r in suite.results],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written: {report_path}")

    sys.exit(0 if (fp + fn) == 0 else 1)


if __name__ == "__main__":
    main()
