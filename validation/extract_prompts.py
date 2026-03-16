#!/usr/bin/env python3
"""
Extract prompts from a test report and save them as a reusable prompts file.

Reads a test_report_*.json produced by test_auto_suite and writes a
saved_prompts_*.json that can be passed to --replay in both:
  • python -m validation.test_auto_suite  --replay <file>
  • python -m validation.run_classifier_check --replay <file>

Extraction rules (from the report's "tests" array):
  tier 2, prompt_used present              → bad_prompts   (adversarial)
  tier 3, "inexact" in detail              → inexact_prompts
  tier 3, prompt_used present, not inexact → good_prompts  (valid car searches)

Duplicate prompts within each bucket are removed; order is preserved.

Usage:
  python -m validation.extract_prompts validation/reports/test_report_XYZ.json
  python -m validation.extract_prompts validation/reports/test_report_XYZ.json --out validation/samples/my_prompts.json
  python -m validation.extract_prompts validation/reports/test_report_XYZ.json --labeled
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dedupe(lst: list) -> list:
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract(report_path: str) -> dict:
    """
    Parse a test report JSON and return a dict with keys:
      good_prompts, bad_prompts, inexact_prompts
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    good, bad, inexact = [], [], []

    for test in report.get("tests", []):
        prompt = test.get("prompt_used")
        if not prompt:
            continue

        tier = test.get("tier")
        detail = (test.get("detail") or "").lower()

        if tier == 2:
            bad.append(prompt)
        elif tier == 3:
            if "inexact" in detail:
                inexact.append(prompt)
            else:
                good.append(prompt)

    return {
        "good_prompts": _dedupe(good),
        "bad_prompts": _dedupe(bad),
        "inexact_prompts": _dedupe(inexact),
    }


def to_labeled(buckets: dict) -> list:
    """Convert saved-prompts buckets to the labeled list format for run_classifier_check."""
    labeled = []
    for p in buckets.get("good_prompts", []):
        labeled.append({"prompt": p, "label": "car"})
    for p in buckets.get("bad_prompts", []):
        labeled.append({"prompt": p, "label": "not_car"})
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract prompts from a test report into a reusable JSON file"
    )
    parser.add_argument(
        "report",
        help="Path to the test_report_*.json file to extract from",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output file path. "
            "Defaults to validation/reports/saved_prompts_<ts>.json"
        ),
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        help=(
            "Write the labeled list format [{prompt, label}] instead of the "
            "saved-prompts dict format. Use this when replaying into "
            "run_classifier_check --replay."
        ),
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"Error: file not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    buckets = extract(str(report_path))

    good_n = len(buckets["good_prompts"])
    bad_n = len(buckets["bad_prompts"])
    inexact_n = len(buckets["inexact_prompts"])
    total = good_n + bad_n + inexact_n

    if total == 0:
        print("No prompts with prompt_used found in this report. Nothing to save.")
        sys.exit(0)

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        reports_dir = ROOT / "validation" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_labeled" if args.labeled else ""
        out_path = reports_dir / f"saved_prompts{suffix}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.labeled:
        labeled = to_labeled(buckets)
        payload = labeled
        print(
            f"Extracted {len(labeled)} labeled prompts "
            f"({good_n} car + {bad_n} not_car) from {report_path.name}"
        )
    else:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "source_report": report_path.name,
            "good_prompts": buckets["good_prompts"],
            "bad_prompts": buckets["bad_prompts"],
            "inexact_prompts": buckets["inexact_prompts"],
        }
        print(
            f"Extracted {total} prompts from {report_path.name}: "
            f"{good_n} good, {bad_n} bad, {inexact_n} inexact"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
