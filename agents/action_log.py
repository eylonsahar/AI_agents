from typing import Any, Dict, List


class ActionLog:
    """
    Tracks every LLM call made by the agent system.

    Each step matches the schema required in the assignment:
        {
            "module": str, top-level module (e.g. "Supervisor", "FieldAgent")
            "submodule": str, # subcomponent (e.g. "DecisionMaking", "MockSeller/GetData")
            "prompt": str, # the full prompt string sent to the LLM
            "response": str # the raw response string returned by the LLM
        }

    Usage:
        log = ActionLog()
        log.add_step("Supervisor", "DecisionMaking", prompt_text, response_text)
        steps = log.get_steps()
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, module: str, submodule: str, prompt: str, response: str) -> None:
        """
        Record a single LLM call.

        Args:
            module:    Top-level module name.
            submodule: Sub-component within that module.
            prompt:    The full prompt sent to the LLM.
            response:  The raw LLM response string.
        """
        self.steps.append({
            "module": module,
            "submodule": submodule,
            "prompt": prompt,
            "response": response
        })

    def get_steps(self) -> List[Dict[str, Any]]:
        """
        Return all logged steps in insertion order.
        """
        return self.steps

    def print_steps(self) -> None:
        """
        Print a human-readable summary of every logged step.
        Can be used for debugging.
        """
        print("\n" + "=" * 80)
        print("📋 AGENT STEP LOG")
        print("=" * 80)

        for i, step in enumerate(self.steps, 1):
            print(f"\n[Step {i}] {step['module']} → {step['submodule']}")

            # Truncate long prompts / responses for console readability
            prompt_preview = step["prompt"]
            if len(prompt_preview) > 200:
                prompt_preview = prompt_preview[:200] + "..."
            print(f"   📤 Prompt:   {prompt_preview}")

            response_preview = step["response"]
            if len(response_preview) > 200:
                response_preview = response_preview[:200] + "..."
            print(f"   📥 Response: {response_preview}")

        print("\n" + "=" * 80)
        print(f"Total LLM calls logged: {len(self.steps)}")
        print("=" * 80 + "\n")
