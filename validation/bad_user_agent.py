"""
BadUserAgent — generates adversarial non-car prompts for Tier 2 classifier tests.
"""

from typing import List

from .base_user_agent import UserAgent


class BadUserAgent(UserAgent):
    """Generates prompts that should NOT be processed by the car-search agent."""

    def generate(self, n: int = 6) -> List[str]:
        """Generate n adversarial prompts via one batch LLM call."""
        if n % 3 != 0:
            print("n must be divisible by 3")
            n=6 # make it divisible by 3
        try:
            prompt = f"""Generate {n} requests that should NOT be processed by a used-car search agent, but each must include a year and a price. Generate a mix:
- {n//3} clearly off-topic (flights, real estate, electronics)
- {n//3} car-adjacent but not car-buying (car insurance, car rental, buying car parts, selling a car)
- {n//3} maximally ambiguous (motorcycles, trucks for work, RVs, boats)
The more the prompt looks like a car search without actually being one, the better. Return one prompt per line, no numbering."""
            out = self._call_llm(prompt)
            return self._parse_lines(out, n)
        except Exception:
            return []
