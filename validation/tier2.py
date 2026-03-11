"""
Tier 2 — Intent classifier tests (adversarial prompts).
Expects status=error with classifier rejection (e.g. "doesn't seem to be about finding a car").
Treats "no vehicles found" as a bypass — pipeline ran instead of rejecting.
"""

import time
import traceback
from typing import List

import requests

from .models import TestResult


def get_step_count(bad_prompts: List[str]) -> int:
    """Return the number of test steps in this tier (for progress bar)."""
    return len(bad_prompts)


def run_tier2(suite, bad_prompts: List[str]) -> None:
    """Tier 2: adversarial prompts — expect status=error, not car search."""
    base = suite.base_url.rstrip("/")
    classifier_bypass_seen = False

    for prompt in bad_prompts:
        if classifier_bypass_seen:
            suite._add_result(
                TestResult(
                    name="tier2_skip_after_bypass",
                    tier=2,
                    passed=True,
                    detail="SKIPPED (budget): classifier bypass already confirmed",
                    prompt_used=prompt,
                )
            )
            suite._advance_progress()
            continue

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
                        response_snapshot=(r.text or "")[:2000] or None,
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
                        response_snapshot=(r.text or "")[:2000] or None,
                    )
                )
                suite._advance_progress()
                continue

            status = j.get("status")
            resp_snap = (j.get("response") or j.get("error") or "")[:2000]
            if status == "ok":
                classifier_bypass_seen = True
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
                    classifier_bypass_seen = True
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
