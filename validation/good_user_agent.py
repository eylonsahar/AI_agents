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

    def generate_inexact(self, n: int = 2) -> List[str]:
        """Generate n prompts that name a plausible-sounding but non-existent make+model.

        These are valid car-search requests (price + year present) but with a
        misspelled or invented model name (e.g. "Toyota Okozaky", "Hunda Civicxs").
        The expected agent behaviour is status=ok WITH a "No exact match found for"
        acknowledgement and alternative vehicles returned.
        """
        if n == 0:
            return []
        try:
            prompt = (
                f"Generate {n} used-car search requests where the vehicle make+model "
                f"sounds like a real car but is slightly misspelled or invented "
                f"(e.g. 'Toyota Okozaky', 'Hunda Civicxs', 'Ford Mustange', 'Chevvy Silverardo'). "
                f"Each request must include a maximum price in USD and a minimum model year. "
                f"Vary the makes and price ranges. "
                f"Return one prompt per line, no numbering."
            )
            out = self._call_llm(prompt)
            return self._parse_lines(out, n)
        except Exception:
            return []
