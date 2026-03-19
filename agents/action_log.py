from typing import Any, Dict, List


class ActionLog:
    """
    Tracks every step made by the agent system.

    Steps include both real LLM calls and deterministic pipeline steps
    (CSV retrieval, scoring, mock-seller calls, etc.).  Use is_llm_call=True
    only for steps that actually invoke the language model.

    Each step matches the schema required in the assignment:
        {
            "module":      str,   top-level module (e.g. "Supervisor", "FieldAgent")
            "submodule":   str,   subcomponent (e.g. "DecisionMaking", "Seller/GetData")
            "prompt":      str,   the full prompt string sent to the LLM (empty for non-LLM steps)
            "response":    str,   the raw response string returned (or deterministic result)
            "is_llm_call": bool,  True only when the step actually called the language model
        }

    Usage:
        log = ActionLog()
        log.add_step("Supervisor", "Thought", prompt_text, response_text, is_llm_call=True)
        log.add_step("SearchPipeline", "ListingsRetriever", "", summary, is_llm_call=False)
        steps = log.get_steps()
        print(log.count_llm_calls(), "real LLM calls")
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        module: str,
        submodule: str,
        prompt: str,
        response: str,
        is_llm_call: bool = True,
    ) -> None:
        """
        Record a single pipeline step.

        Args:
            module:      Top-level module name.
            submodule:   Sub-component within that module.
            prompt:      The full prompt sent to the LLM (empty string for non-LLM steps).
            response:    The raw response / deterministic result string.
            is_llm_call: True when this step actually called the language model.
        """
        self.steps.append({
            "module": module,
            "submodule": submodule,
            "prompt": prompt,
            "response": response,
            "is_llm_call": is_llm_call,
        })

    def get_steps(self) -> List[Dict[str, Any]]:
        """Return all logged steps in insertion order."""
        return self.steps

    def count_llm_calls(self) -> int:
        """Return the number of steps that actually invoked the language model."""
        return sum(1 for s in self.steps if s.get("is_llm_call", True))

    def print_steps(self) -> None:
        """Print a human-readable summary of every logged step."""
        print("\n" + "=" * 80)
        print("📋 AGENT STEP LOG")
        print("=" * 80)

        for i, step in enumerate(self.steps, 1):
            print(f"\n[Step {i}] {step['module']} → {step['submodule']}")

            prompt_preview = step["prompt"]
            if len(prompt_preview) > 200:
                prompt_preview = prompt_preview[:200] + "..."
            print(f"   📤 Prompt:   {prompt_preview}")

            response_preview = step["response"]
            if len(response_preview) > 200:
                response_preview = response_preview[:200] + "..."
            print(f"   📥 Response: {response_preview}")

        print("\n" + "=" * 80)
        print(f"Total steps logged:     {len(self.steps)}")
        print(f"Real LLM calls:         {self.count_llm_calls()}")
        print("=" * 80 + "\n")
