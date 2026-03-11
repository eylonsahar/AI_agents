"""
Tier 1 — API contract tests (server needed, no LLM).
GET endpoints, agent_info completeness, frontend smoke, input validation.
"""

import time
import traceback

import requests

from .models import TestResult

# 5 GET tests + validation_cases + 1 invalid_json
NUM_GET_TESTS = 5
VALIDATION_CASES = [
    ("test_missing_prompt_field", {}, 422, "prompt"),
    ("test_null_prompt", {"prompt": None}, 422, None),
    ("test_empty_prompt", {"prompt": ""}, 200, "cannot be empty"),
    ("test_whitespace_only", {"prompt": "   "}, 200, "cannot be empty"),
    ("test_short_prompt", {"prompt": "car"}, 200, "too short"),
    ("test_no_budget", {"prompt": "nice family sedan 2018 or newer"}, 200, "budget"),
    ("test_no_year", {"prompt": "family sedan under $20000"}, 200, "year"),
    ("test_extra_fields_ignored", {"prompt": "", "unknown_field": 99}, 200, "cannot be empty"),
]


def get_step_count() -> int:
    """Return the number of test steps in this tier (for progress bar)."""
    return NUM_GET_TESTS + len(VALIDATION_CASES) + 1


def run_tier1(suite) -> None:
    """Tier 1: GET endpoints, agent_info completeness, frontend, input validation."""
    base = suite.base_url.rstrip("/")

    # test_team_info
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/api/team_info", timeout=10)
        dur = (time.perf_counter() - t0) * 1000
        passed = r.status_code == 200
        if passed:
            d = r.json()
            passed = (
                "group_batch_order_number" in d
                and "team_name" in d
                and "students" in d
                and isinstance(d["students"], list)
                and len(d["students"]) == 3
                and all("name" in s and "email" in s for s in d["students"])
            )
        suite._add_result(
            TestResult(
                name="test_team_info",
                tier=1,
                passed=passed,
                detail=f"status={r.status_code}, body ok" if passed else f"status={r.status_code} or schema invalid",
                duration_ms=dur,
            )
        )
    except Exception as e:
        suite._add_result(
            TestResult(
                name="test_team_info",
                tier=1,
                passed=False,
                detail=str(e),
                duration_ms=0,
                exception_traceback=traceback.format_exc(),
                request_snapshot={"url": f"{base}/api/team_info"},
            )
        )
    suite._advance_progress()

    # test_agent_info_completeness
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/api/agent_info", timeout=10)
        dur = (time.perf_counter() - t0) * 1000
        passed = r.status_code == 200
        if passed:
            d = r.json()
            passed = (
                d.get("description")
                and d.get("purpose")
                and "prompt_template" in d
                and "template" in d.get("prompt_template", {})
                and d.get("prompt_examples")
            )
            if passed:
                for ex in d["prompt_examples"]:
                    if not (ex.get("prompt") and ex.get("full_response") and ex.get("steps")):
                        passed = False
                        break
                    for step in ex.get("steps", []):
                        if not all(k in step for k in ("module", "prompt", "response")):
                            passed = False
                            break
        suite._add_result(
            TestResult(
                name="test_agent_info_completeness",
                tier=1,
                passed=passed,
                detail="ok" if passed else "missing required fields or invalid schema",
                duration_ms=dur,
            )
        )
        if passed:
            suite._declared_modules = {
                s.get("module") for ex in d.get("prompt_examples", []) for s in ex.get("steps", [])
            }
            suite._declared_modules.discard(None)
            suite._declared_modules.discard("")
    except Exception as e:
        suite._add_result(
            TestResult(
                name="test_agent_info_completeness",
                tier=1,
                passed=False,
                detail=str(e),
                duration_ms=0,
                exception_traceback=traceback.format_exc(),
                request_snapshot={"url": f"{base}/api/agent_info"},
            )
        )
    suite._advance_progress()

    # test_model_architecture
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/api/model_architecture", timeout=10)
        dur = (time.perf_counter() - t0) * 1000
        ct = r.headers.get("Content-Type", "")
        passed = r.status_code == 200 and "image/png" in ct and len(r.content) > 0
        suite._add_result(
            TestResult(
                name="test_model_architecture",
                tier=1,
                passed=passed,
                detail=f"status={r.status_code}, Content-Type={ct}, len={len(r.content)}",
                duration_ms=dur,
            )
        )
    except Exception as e:
        suite._add_result(
            TestResult(
                name="test_model_architecture",
                tier=1,
                passed=False,
                detail=str(e),
                duration_ms=0,
                exception_traceback=traceback.format_exc(),
                request_snapshot={"url": f"{base}/api/model_architecture"},
            )
        )
    suite._advance_progress()

    # test_frontend_smoke
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/", timeout=10)
        dur = (time.perf_counter() - t0) * 1000
        ct = r.headers.get("Content-Type", "")
        body = r.text if hasattr(r, "text") else r.content.decode(errors="replace")
        passed = (
            r.status_code == 200
            and "text/html" in ct
            and "<textarea" in body
            and "Run Agent" in body
            and "api/execute" in body
        )
        suite._add_result(
            TestResult(
                name="test_frontend_smoke",
                tier=1,
                passed=passed,
                detail="ok" if passed else f"status={r.status_code}, html checks failed",
                duration_ms=dur,
            )
        )
    except Exception as e:
        suite._add_result(
            TestResult(
                name="test_frontend_smoke",
                tier=1,
                passed=False,
                detail=str(e),
                duration_ms=0,
                exception_traceback=traceback.format_exc(),
                request_snapshot={"url": f"{base}/"},
            )
        )
    suite._advance_progress()

    # test_unknown_route
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/api/nonexistent", timeout=5)
        dur = (time.perf_counter() - t0) * 1000
        passed = r.status_code != 500
        suite._add_result(
            TestResult(
                name="test_unknown_route",
                tier=1,
                passed=passed,
                detail=f"status={r.status_code}",
                duration_ms=dur,
            )
        )
    except Exception as e:
        suite._add_result(
            TestResult(
                name="test_unknown_route",
                tier=1,
                passed=False,
                detail=str(e),
                duration_ms=0,
                exception_traceback=traceback.format_exc(),
                request_snapshot={"url": f"{base}/api/nonexistent"},
            )
        )
    suite._advance_progress()

    # Input validation tests
    for name, body, exp_status, exp_in_error in VALIDATION_CASES:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base}/api/execute", json=body, timeout=10)
            dur = (time.perf_counter() - t0) * 1000
            passed = r.status_code != 500
            if exp_status is not None:
                passed = passed and r.status_code == exp_status
            if exp_in_error:
                try:
                    j = r.json()
                    err = j.get("error") or j.get("detail") or ""
                    if isinstance(err, list):
                        err = str(err)
                    err = str(err).lower()
                    passed = passed and exp_in_error.lower() in err
                except Exception:
                    passed = False
            suite._add_result(
                TestResult(
                    name=name,
                    tier=1,
                    passed=passed,
                    detail=f"status={r.status_code}",
                    duration_ms=dur,
                )
            )
        except Exception as e:
            suite._add_result(
                TestResult(
                    name=name,
                    tier=1,
                    passed=False,
                    detail=str(e),
                    duration_ms=0,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"url": f"{base}/api/execute", "body": body},
                )
            )
        suite._advance_progress()

    # test_invalid_json
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base}/api/execute",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        dur = (time.perf_counter() - t0) * 1000
        passed = r.status_code == 422
        suite._add_result(
            TestResult(
                name="test_invalid_json",
                tier=1,
                passed=passed,
                detail=f"status={r.status_code}",
                duration_ms=dur,
            )
        )
    except Exception as e:
        suite._add_result(
            TestResult(
                name="test_invalid_json",
                tier=1,
                passed=False,
                detail=str(e),
                duration_ms=0,
                exception_traceback=traceback.format_exc(),
                request_snapshot={"url": f"{base}/api/execute", "body": "not json"},
            )
        )
    suite._advance_progress()
