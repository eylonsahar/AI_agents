"""
Abstract base class for prompt-generation agents used in validation.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from .models import ROOT


class UserAgent(ABC):
    """Abstract base for agents that generate prompts for validation testing."""

    def __init__(self) -> None:
        self._llm = None

    def _get_llm(self):
        """Lazily initialize the LLM gateway."""
        if self._llm is None:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            from gateways.llm_gateway import LLMGateway

            self._llm = LLMGateway.get_instance(api_key=api_key)
        return self._llm

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt and return the response text."""
        gw = self._get_llm()
        out, _ = gw.call_llm(prompt=prompt)
        return out.strip()

    def _parse_lines(self, text: str, n: int) -> List[str]:
        """Parse LLM output into a list of non-empty lines, capped at n."""
        lines = [s.strip() for s in text.split("\n") if s.strip()]
        return lines[:n]

    @abstractmethod
    def generate(self, n: int = 6) -> List[str]:
        """Generate n prompts. Return a list of prompt strings."""
        pass
