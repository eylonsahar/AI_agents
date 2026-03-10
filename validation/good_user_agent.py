"""
GoodUserAgent — generates valid car-search prompts for Tier 3 pipeline tests.
"""

from typing import List

from .base_user_agent import UserAgent


class GoodUserAgent(UserAgent):
    """Generates diverse, realistic used-car search requests."""

    def generate(self, n: int = 6) -> List[str]:
        """Generate n valid car-search prompts via one batch LLM call."""
        try:
            prompt = f"""Generate {n} diverse, realistic used-car search requests. Each must contain a maximum price in USD and a minimum model year. Vary vehicle type, budget range, and phrasing. Return one prompt per line, no numbering."""
            out = self._call_llm(prompt)
            return self._parse_lines(out, n)
        except Exception:
            return []
