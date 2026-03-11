"""
AItzik validation package — automated test suite for pre-deployment verification.

Tiers: 0 (unit), 1 (API), 2 (adversarial), 3 (pipeline).
Agents: GoodUserAgent, BadUserAgent (inherit from UserAgent).
"""

from .base_user_agent import UserAgent
from .bad_user_agent import BadUserAgent
from .good_user_agent import GoodUserAgent
from .models import TestResult
from .test_auto_suite import AItzikTestSuite

__all__ = [
    "AItzikTestSuite",
    "BadUserAgent",
    "GoodUserAgent",
    "TestResult",
    "UserAgent",
]
