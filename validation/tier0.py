"""
Tier 0 — Pure unit tests (no server, no LLM).
Tests _extract_price and _extract_year from api.server.
"""

import time
import traceback

from .models import TestResult

PRICE_CASES = [
    ("SUV under $20,000 from 2018", "20000"),
    ("budget 15k, 2019 or newer", "15000"),
    ("max price 22000, year 2020", "22000"),
    ("under $1,500 from 2021", "1500"),
    ("car for 25,000 dollars", "25000"),
    ("car from 1999 under 20000", "20000"),
    ("I want a 2020 sedan for $500", "500"),
    ("20k budget from 2018", "20000"),
    ("sedan, $30000, 2021 or newer", "30000"),
    ("price is about 18000 dollars", "18000"),
    ("nice car under $10000 from 2019", "10000"),
    ("$1200 used car from 2020", "1200"),
    # Fix 1: under 2000 (2000 allowed; 2001-2099 stay excluded as year-like)
    ("under 2000 corola from 2000", "2000"),
    ("under 2019", None),  # 2019 excluded—year-like
    # Fix 2: decimal k-suffix
    ("1.5k budget", "1500"),
    ("car from 2020 2.5k max", "2500"),
    # Fix 3: comma-separated without $ sign
    ("under 2,500", "2500"),
    ("below 15,000 from 2018", "15000"),
]

YEAR_CASES = [
    ("from 2018, SUV under $20k", "2018"),
    ("2019 or newer, budget $15k", "2019"),
    ("min year 2020 sedan", "2020"),
    ("since 2017, budget $15k", "2017"),
    ("car from 1999 under 20000", "1999"),
    ("I want a 2020 model for $15000", "2020"),
    ("nice car under $10000", None),
    ("car under $20000 from the 90s", None),
    ("2025 or newer electric car under $40000", "2025"),
    ("reliable truck, budget $25000, year minimum 2016", "2016"),
]


def get_step_count() -> int:
    """Return the number of test steps in this tier (for progress bar)."""
    return len(PRICE_CASES) + len(YEAR_CASES)


def run_tier0(suite) -> None:
    """Tier 0: extract_price and extract_year unit tests."""
    from api.server import _extract_price, _extract_year

    for prompt, expected in PRICE_CASES:
        t0 = time.perf_counter()
        try:
            got = _extract_price(prompt)
            dur = (time.perf_counter() - t0) * 1000
            passed = got == expected
            suite._add_result(
                TestResult(
                    name=f"extract_price: {prompt[:40]}...",
                    tier=0,
                    passed=passed,
                    detail=f"expected {expected!r}, got {got!r}" if not passed else "ok",
                    duration_ms=dur,
                )
            )
        except Exception as e:
            suite._add_result(
                TestResult(
                    name=f"extract_price: {prompt[:40]}...",
                    tier=0,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": prompt, "expected": expected},
                )
            )
        suite._advance_progress()

    for prompt, expected in YEAR_CASES:
        t0 = time.perf_counter()
        try:
            got = _extract_year(prompt)
            dur = (time.perf_counter() - t0) * 1000
            passed = got == expected
            suite._add_result(
                TestResult(
                    name=f"extract_year: {prompt[:40]}...",
                    tier=0,
                    passed=passed,
                    detail=f"expected {expected!r}, got {got!r}" if not passed else "ok",
                    duration_ms=dur,
                )
            )
        except Exception as e:
            suite._add_result(
                TestResult(
                    name=f"extract_year: {prompt[:40]}...",
                    tier=0,
                    passed=False,
                    detail=str(e),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    exception_traceback=traceback.format_exc(),
                    request_snapshot={"prompt": prompt, "expected": expected},
                )
            )
        suite._advance_progress()
