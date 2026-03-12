"""
Tier 3 — Full pipeline tests (budget-guarded).
Fixed edge cases + optional random prompts from GoodUserAgent.
"""

import json
import time
import traceback
from typing import List, Tuple

import requests

from .models import EFFICIENCY_STEP_THRESHOLD, TestResult

FIXED_CASES = [
    ("spec_compliant_prompt_only", {"prompt": "Reliable family SUV under $22,000, 2018 or newer"}, None),
    ("explicit_fields", {"prompt": "family SUV", "max_price": "22000", "year_min": "2018"}, None),
    # "inexact" = status must be "ok" AND response must contain "No exact match found for"
    ("inexact_model", {"prompt": "Toyota Okozaky under $50000, 2020 or newer"}, "inexact"),
]


def get_fixed_step_count() -> int:
    """Return the number of fixed test steps (fixed cases + streaming slot) for progress bar."""
    return len(FIXED_CASES) + 1  # +1 for streaming


def validate_response_schema(j: dict) -> Tuple[bool, str]:
    """Check PDF-required schema: status, error, response, steps; each step has module, prompt, response."""
    if "status" not in j:
        return False, "missing status"
    if "error" not in j:
        return False, "missing error"
    if "response" not in j:
        return False, "missing response"
    if "steps" not in j or not isinstance(j["steps"], list):
        return False, "missing or invalid steps"
    for i, s in enumerate(j["steps"]):
        if not isinstance(s, dict):
            return False, f"step {i} not dict"
        if "module" not in s or "prompt" not in s or "response" not in s:
            return False, f"step {i} missing module/prompt/response"
    return True, "ok"


def run_tier3(suite, good_prompts: List[str], inexact_prompts: List[str] = None) -> None:
    """Tier 3: full pipeline tests with fixed edge cases + optional random prompts."""
    if inexact_prompts is None:
        inexact_prompts = []
    base = suite.base_url.rstrip("/")

    for name, body, expect_error in FIXED_CASES:
        if not suite._can_run_pipeline():
            suite._add_result(
                TestResult(
                    name=f"tier3_{name}",
                    tier=3,
                    passed=True,
                    detail="SKIPPED (budget exhausted)",
                    prompt_used=body.get("prompt", ""),
                )
            )
            suite._advance_progress()
            continue

        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base}/api/execute", json=body, timeout=600)
            dur = (time.perf_counter() - t0) * 1000
            suite.pipeline_runs_used += 1

            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            ok_schema, schema_msg = validate_response_schema(j)
            resp_snap = (j.get("response") or j.get("error") or "")[:2000]
            if not ok_schema:
                suite._add_result(
                    TestResult(
                        name=f"tier3_{name}",
                        tier=3,
                        passed=False,
                        detail=f"schema invalid: {schema_msg}",
                        duration_ms=dur,
                        prompt_used=body.get("prompt", ""),
                        response_snapshot=resp_snap or None,
                    )
                )
                suite._advance_progress()
                continue

            status = j.get("status", "")
            if expect_error == "inexact":
                # Pass only when the agent returned cars AND acknowledged the missing model
                response_text = (j.get("response") or "").lower()
                passed = (
                    status == "ok"
                    and "no exact match found for" in response_text
                )
                suite._add_result(
                    TestResult(
                        name=f"tier3_{name}",
                        tier=3,
                        passed=passed,
                        detail=(
                            f"status={status}, "
                            f"inexact_banner={'present' if 'no exact match found for' in response_text else 'MISSING'}"
                        ),
                        duration_ms=dur,
                        prompt_used=body.get("prompt", ""),
                        response_snapshot=resp_snap or None,
                    )
                )
                suite._advance_progress()
            elif expect_error:
                err = (j.get("error") or "").lower()
                passed = status == "error" and ("no vehicles" in err or "not find" in err or "couldn't find" in err or ". stopping." in err)
                suite._add_result(
                    TestResult(
                        name=f"tier3_{name}",
                        tier=3,
                        passed=passed,
                        detail=f"expected error, got status={status}",
                        duration_ms=dur,
                        prompt_used=body.get("prompt", ""),
                        response_snapshot=resp_snap or None,
                    )
                )
                suite._advance_progress()
            else:
                passed = status == "ok"
                step_count = len(j.get("steps", []))
                suite.efficiency.append(
                    {
                        "prompt": body.get("prompt", ""),
                        "step_count": step_count,
                        "flagged": step_count > EFFICIENCY_STEP_THRESHOLD,
                    }
                )
                if passed:
                    actual_modules = {s.get("module", "") for s in j.get("steps", [])}
                    actual_modules.discard("")
                    if hasattr(suite, "_declared_modules") and suite._declared_modules:
                        suite._module_consistency = {
                            "declared": list(suite._declared_modules),
                            "actual": list(actual_modules),
                            "mismatches": list(suite._declared_modules ^ actual_modules),
                        }
                suite._add_result(
                    TestResult(
                        name=f"tier3_{name}",
                        tier=3,
                        passed=passed,
                        detail=f"status={status}, steps={step_count}",
                        duration_ms=dur,
                        prompt_used=body.get("prompt", ""),
                        response_snapshot=resp_snap or None,
                    )
                )
                suite._advance_progress()
        except Exception as e:
            suite._add_result(
                TestResult(
                    name=f"tier3_{name}",
                    tier=3,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    prompt_used=body.get("prompt", ""),
                    exception_traceback=traceback.format_exc(),
                    request_snapshot=body,
                )
            )
            suite._advance_progress()

    # Streaming path
    if suite._can_run_pipeline():
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{base}/api/execute",
                json={"prompt": "Reliable family SUV under $22,000, 2018 or newer"},
                params={"stream": "true"},
                timeout=600,
                stream=True,
            )
            lines = []
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    lines.append(line)
            dur = (time.perf_counter() - t0) * 1000
            suite.pipeline_runs_used += 1

            seen_processing = False
            seen_ok = False
            final_status = None
            for line in lines:
                try:
                    d = json.loads(line)
                    s = d.get("status", "")
                    if s == "processing":
                        seen_processing = True
                    elif s == "ok":
                        seen_ok = True
                        final_status = d
                    elif s == "error":
                        final_status = d
                except Exception:
                    pass

            passed = seen_processing and (seen_ok or final_status)
            if final_status and final_status.get("status") == "ok":
                passed = passed and validate_response_schema(final_status)[0]
            resp_snap = None
            if final_status:
                resp_snap = (final_status.get("response") or final_status.get("error") or "")[:2000]
            suite._add_result(
                TestResult(
                    name="tier3_streaming",
                    tier=3,
                    passed=passed,
                    detail=f"processing={seen_processing}, ok={seen_ok}, lines={len(lines)}",
                    duration_ms=dur,
                    prompt_used="Reliable family SUV under $22,000, 2018 or newer",
                    response_snapshot=resp_snap,
                )
            )
            suite._advance_progress()
        except Exception as e:
            suite._add_result(
                TestResult(
                    name="tier3_streaming",
                    tier=3,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    prompt_used="Reliable family SUV under $22,000, 2018 or newer",
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": "Reliable family SUV under $22,000, 2018 or newer", "stream": True},
                )
            )
            suite._advance_progress()
    else:
        suite._advance_progress()  # streaming slot skipped (budget exhausted)

    # Random good prompts
    for prompt in good_prompts:
        if not suite._can_run_pipeline():
            suite._add_result(
                TestResult(
                    name="tier3_random",
                    tier=3,
                    passed=True,
                    detail="SKIPPED (budget exhausted)",
                    prompt_used=prompt,
                )
            )
            suite._advance_progress()
            continue

        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base}/api/execute", json={"prompt": prompt}, timeout=600)
            dur = (time.perf_counter() - t0) * 1000
            suite.pipeline_runs_used += 1

            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            ok_schema, _ = validate_response_schema(j)
            status = j.get("status", "")
            step_count = len(j.get("steps", []))
            suite.efficiency.append(
                {"prompt": prompt[:60], "step_count": step_count, "flagged": step_count > EFFICIENCY_STEP_THRESHOLD}
            )
            resp_snap = (j.get("response") or j.get("error") or "")[:2000]

            passed = ok_schema and status == "ok"
            suite._add_result(
                TestResult(
                    name="tier3_random",
                    tier=3,
                    passed=passed,
                    detail=f"status={status}, steps={step_count}",
                    duration_ms=dur,
                    prompt_used=prompt,
                    response_snapshot=resp_snap or None,
                )
            )
            suite._advance_progress()
        except Exception as e:
            suite._add_result(
                TestResult(
                    name="tier3_random",
                    tier=3,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    prompt_used=prompt,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": prompt},
                )
            )
            suite._advance_progress()

    # Inexact-model prompts — must return ok + "No exact match found for" banner
    for prompt in inexact_prompts:
        if not suite._can_run_pipeline():
            suite._add_result(
                TestResult(
                    name="tier3_inexact_model",
                    tier=3,
                    passed=True,
                    detail="SKIPPED (budget exhausted)",
                    prompt_used=prompt,
                )
            )
            suite._advance_progress()
            continue

        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base}/api/execute", json={"prompt": prompt}, timeout=600)
            dur = (time.perf_counter() - t0) * 1000
            suite.pipeline_runs_used += 1

            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            ok_schema, schema_msg = validate_response_schema(j)
            status = j.get("status", "")
            response_text = (j.get("response") or "").lower()
            resp_snap = (j.get("response") or j.get("error") or "")[:2000]

            if not ok_schema:
                passed = False
                detail = f"schema invalid: {schema_msg}"
            else:
                inexact_banner_present = "no exact match found for" in response_text
                passed = status == "ok" and inexact_banner_present
                detail = (
                    f"status={status}, "
                    f"inexact_banner={'present' if inexact_banner_present else 'MISSING'}"
                )

            suite._add_result(
                TestResult(
                    name="tier3_inexact_model",
                    tier=3,
                    passed=passed,
                    detail=detail,
                    duration_ms=dur,
                    prompt_used=prompt,
                    response_snapshot=resp_snap or None,
                )
            )
            suite._advance_progress()
        except Exception as e:
            suite._add_result(
                TestResult(
                    name="tier3_inexact_model",
                    tier=3,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    prompt_used=prompt,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": prompt},
                )
            )
            suite._advance_progress()
