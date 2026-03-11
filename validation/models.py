"""
Shared models and constants for the validation suite.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
COST_PER_RUN_ESTIMATE = 0.03
EFFICIENCY_STEP_THRESHOLD = 40


@dataclass
class TestResult:
    name: str
    tier: int
    passed: bool
    detail: str
    duration_ms: float = 0.0
    prompt_used: Optional[str] = None
    response_snapshot: Optional[str] = None  # AItzik's final response text (Tier 2/3)
    exception_traceback: Optional[str] = None  # Full traceback when unexpected exception occurs
    request_snapshot: Optional[dict] = None  # Request that caused failure (for replay)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prompt_used"] = self.prompt_used
        d["response_snapshot"] = self.response_snapshot
        d["exception_traceback"] = self.exception_traceback
        d["request_snapshot"] = self.request_snapshot
        return d
