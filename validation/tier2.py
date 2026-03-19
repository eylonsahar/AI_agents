"""
Tier 2 — Intent classifier tests (adversarial prompts).

Full mode  (run_tier2):             POST /api/execute — full pipeline, budget-guarded.
Short mode (run_tier2_classifier_only): POST /api/classify — classifier only, no pipeline cost.
"""

import time
import traceback
from typing import Dict, List

import requests

from .models import TestResult


def get_step_count(bad_prompts: List[str]) -> int:
    """Return the number of test steps in this tier (for progress bar)."""
    return len(bad_prompts)


def run_tier2_classifier_only(suite, labeled_prompts: List[Dict[str, str]]) -> None:
    """
    Classifier-only short run: POST /api/classify for each labeled prompt.
    No pipeline budget is consumed.

    Each entry must be {"prompt": str, "label": "car" | "not_car"}.

      label=car     → expect is_car_search=true
                        PASS = True Positive (TP)
                        FAIL = False Negative (FN) — classifier wrongly rejected a car prompt

      label=not_car → expect is_car_search=false
                        PASS = True Negative (TN)
                        FAIL = False Positive (FP) — classifier let a non-car prompt through
    """
    base = suite.base_url.rstrip("/")

    for entry in labeled_prompts:
        prompt = entry["prompt"]
        label = entry["label"]          # "car" or "not_car"
        expect_car = label == "car"

        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{base}/api/classify",
                json={"prompt": prompt},
                timeout=60,
            )
            dur = (time.perf_counter() - t0) * 1000

            is_json = r.headers.get("content-type", "").startswith("application/json")
            if not is_json:
                suite._add_result(
                    TestResult(
                        name="tier2c_invalid_response",
                        tier=2,
                        passed=False,
                        detail=f"expected JSON response, got: {r.headers.get('content-type', '') or 'none'}",
                        duration_ms=dur,
                        prompt_used=prompt,
                        response_snapshot=r.text or None,
                    )
                )
                suite._advance_progress()
                continue

            try:
                j = r.json()
            except Exception as parse_err:
                suite._add_result(
                    TestResult(
                        name="tier2c_invalid_json",
                        tier=2,
                        passed=False,
                        detail=f"response is not valid JSON: {parse_err}",
                        duration_ms=dur,
                        prompt_used=prompt,
                        response_snapshot=r.text or None,
                    )
                )
                suite._advance_progress()
                continue

            is_car = j.get("is_car_search")
            if not isinstance(is_car, bool):
                suite._add_result(
                    TestResult(
                        name="tier2c_unexpected_response",
                        tier=2,
                        passed=False,
                        detail=f"unexpected /api/classify response: {j}",
                        duration_ms=dur,
                        prompt_used=prompt,
                    )
                )
                suite._advance_progress()
                continue

            correct = is_car == expect_car
            if expect_car:
                outcome = "TP" if correct else "FN"
                name = "tier2c_accepted" if correct else "tier2c_false_negative"
                detail = (
                    "correctly accepted by classifier (TP)"
                    if correct
                    else "false negative: classifier rejected a valid car prompt (FN)"
                )
            else:
                outcome = "TN" if correct else "FP"
                name = "tier2c_rejected" if correct else "tier2c_false_positive"
                detail = (
                    "correctly rejected by classifier (TN)"
                    if correct
                    else "false positive: classifier accepted a non-car prompt (FP)"
                )

            suite._add_result(
                TestResult(
                    name=name,
                    tier=2,
                    passed=correct,
                    detail=f"[label={label}] [{outcome}] {detail}",
                    duration_ms=dur,
                    prompt_used=prompt,
                )
            )
            suite._advance_progress()
        except Exception as e:
            suite._add_result(
                TestResult(
                    name="tier2c_error",
                    tier=2,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    prompt_used=prompt,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": prompt},
                )
            )
            suite._advance_progress()


def run_tier2(suite, bad_prompts: List[str]) -> None:
    """Tier 2: adversarial prompts — expect status=error, not car search."""
    base = suite.base_url.rstrip("/")

    for prompt in bad_prompts:
        if not suite._can_run_pipeline():
            suite._add_result(
                TestResult(
                    name="tier2_skipped_budget",
                    tier=2,
                    passed=True,
                    detail="SKIPPED (budget exhausted)",
                    prompt_used=prompt,
                )
            )
            suite._advance_progress()
            continue

        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{base}/api/execute",
                json={"prompt": prompt},
                timeout=600,
            )
            dur = (time.perf_counter() - t0) * 1000
            suite.pipeline_runs_used += 1

            is_json = r.headers.get("content-type", "").startswith("application/json")
            if not is_json:
                suite._add_result(
                    TestResult(
                        name="tier2_invalid_response",
                        tier=2,
                        passed=False,
                        detail=f"expected JSON response, got content-type: {r.headers.get('content-type', '') or 'none'}",
                        duration_ms=dur,
                        prompt_used=prompt,
                        response_snapshot=r.text or None,
                    )
                )
                suite._advance_progress()
                continue

            try:
                j = r.json()
            except Exception as parse_err:
                suite._add_result(
                    TestResult(
                        name="tier2_invalid_json",
                        tier=2,
                        passed=False,
                        detail=f"response is not valid JSON: {parse_err}",
                        duration_ms=dur,
                        prompt_used=prompt,
                        response_snapshot=r.text or None,
                    )
                )
                suite._advance_progress()
                continue

            status = j.get("status")
            resp_snap = j.get("response") or j.get("error") or ""
            if status == "ok":
                suite._add_result(
                    TestResult(
                        name="tier2_classifier_bypass",
                        tier=2,
                        passed=False,
                        detail="classifier bypass: pipeline ran on non-car prompt",
                        duration_ms=dur,
                        prompt_used=prompt,
                        response_snapshot=resp_snap or None,
                    )
                )
            elif status == "error":
                err = j.get("error", "") or ""
                # "No vehicles found" means the pipeline ran — classifier failed to reject
                if "no vehicles found" in err.lower() or "couldn't find any vehicles" in err.lower():
                    suite._add_result(
                        TestResult(
                            name="tier2_classifier_bypass",
                            tier=2,
                            passed=False,
                            detail="classifier bypass: pipeline ran (no vehicles found) instead of rejecting non-car prompt",
                            duration_ms=dur,
                            prompt_used=prompt,
                            response_snapshot=resp_snap or None,
                        )
                    )
                else:
                    suite._add_result(
                        TestResult(
                            name="tier2_rejected",
                            tier=2,
                            passed=True,
                            detail=f"correctly rejected: {err[:80]}",
                            duration_ms=dur,
                            prompt_used=prompt,
                            response_snapshot=resp_snap or None,
                        )
                    )
            else:
                suite._add_result(
                    TestResult(
                        name="tier2_unexpected_status",
                        tier=2,
                        passed=False,
                        detail=f"expected status 'error' for rejection, got '{status}' (missing or unexpected)",
                        duration_ms=dur,
                        prompt_used=prompt,
                        response_snapshot=resp_snap or None,
                    )
                )
            suite._advance_progress()
        except Exception as e:
            suite._add_result(
                TestResult(
                    name="tier2_error",
                    tier=2,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    prompt_used=prompt,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": prompt},
                )
            )
            suite._advance_progress()
