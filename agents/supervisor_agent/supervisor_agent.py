from typing import Dict, Any, Optional
from agents.supervisor_agent.user_communication import UserCommunication
from agents.search_agents.listings_retriever import retrieve_listings_from_csv
from agents.field_agent.field_agent import FieldAgent
from gateways.llm_gateway import LLMGateway
from agents.action_log import ActionLog
from dotenv import load_dotenv
import os
import json
from agents.prompts import SUPERVISOR_DECISION_PROMPT
from config import MAX_DECISION_ITERATIONS, NUM_TARGET_LISTINGS, MAX_RECOMMENDED_VEHICLES

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ============================================================================
# Autonomous Supervisor Agent
# ============================================================================
class AgentSupervisor:
    """
    Autonomous supervisor that coordinates the car-finding mission.

    Logging contract:
        - self.action_log is the single ActionLog instance for the entire run.
        - Every LLM call made BY this class is logged here directly.
        - When FieldAgent runs, it receives this same ActionLog and appends
          its own steps to it.  No concatenation needed at the end.
    """

    def __init__(
            self,
            llm_gateway: LLMGateway,
            rag_result_json_path: str = "agents/search_agents/test_result2_suv.json",
            target_listings: int = NUM_TARGET_LISTINGS,
            max_iterations: int = MAX_DECISION_ITERATIONS
    ):
        self.llm_gateway = llm_gateway
        self.rag_result_json_path = rag_result_json_path
        self.user_comm = UserCommunication()

        self.target_listings = target_listings
        self.max_iterations = max_iterations

        # Shared action log — passed down to FieldAgent
        self.action_log = ActionLog()

        # State tracking
        self.user_requirements = None
        self.vehicle_models = None       # will be populated by the stub (or RAG later)
        self.listings = None
        self.processed_results = None
        self.iteration_count = 0

        # Execution tracking
        self.actions_taken = set()

    # ========================================================================
    # State Management
    # ========================================================================
    def _get_current_state(self) -> str:
        """Generate a state summary for the LLM decision prompt."""

        state_parts = [
            f"MISSION: Find {self.target_listings} car listings",
            f"ITERATION: {self.iteration_count}/{self.max_iterations}",
            ""
        ]

        # User requirements
        if self.user_requirements:
            state_parts.append("USER REQUIREMENTS: Collected")
            state_parts.append(f"   - Max Price: ${self.user_requirements.get('max_price', 'N/A')}")
            state_parts.append(f"   - Min Year: {self.user_requirements.get('year_min', 'N/A')}")
            state_parts.append(f"   - Description: {self.user_requirements.get('description', 'N/A')}")
        else:
            state_parts.append("USER REQUIREMENTS: Not collected")

        state_parts.append("")

        # Vehicle models (stub output or future RAG output)
        if self.vehicle_models:
            state_parts.append(f"VEHICLE MODELS: {len(self.vehicle_models)} found")
            for v in self.vehicle_models[:MAX_RECOMMENDED_VEHICLES]:
                state_parts.append(f"   - {v.get('make', '')} {v.get('model', '')}")
        else:
            state_parts.append("VEHICLE MODELS: Not searched")

        state_parts.append("")

        # Listings
        if self.listings:
            total = sum(len(r.get("listings", [])) for r in self.listings.get("results", []))
            state_parts.append(f"LISTINGS: {total} retrieved")
        else:
            state_parts.append("LISTINGS: Not retrieved")

        state_parts.append("")

        # Processing
        if self.processed_results:
            stats = self.processed_results.get("stats", {})
            state_parts.append("PROCESSING: Complete")
            state_parts.append(f"   - Completed: {stats.get('completed_listings', 0)}")
            state_parts.append(f"   - Decisions made: {stats.get('total_decisions', 0)}")
        else:
            state_parts.append("PROCESSING: Not started")

        state_parts.append("")
        state_parts.append(f"ACTIONS TAKEN: {', '.join(self.actions_taken) if self.actions_taken else 'None'}")

        return "\n".join(state_parts)

    def _evaluate_mission_success(self) -> bool:
        """Check if enough listings have been fully processed."""
        if not self.processed_results:
            return False
        completed = self.processed_results.get("stats", {}).get("completed_listings", 0)
        return completed >= self.target_listings

    # ========================================================================
    # Decision Making
    # ========================================================================
    def _make_strategic_decision(self) -> Optional[Dict[str, Any]]:
        """
        Call LLM to decide the next action.  Logs both prompt and response
        to self.action_log before returning the parsed decision.
        """
        state = self._get_current_state()
        prompt = SUPERVISOR_DECISION_PROMPT.format(
            target_listings=self.target_listings,
            state=state,
            MAX_RECOMMENDED_VEHICLES=MAX_RECOMMENDED_VEHICLES
        )

        try:
            response, _ = self.llm_gateway.call_llm(prompt=prompt)

            # Log this LLM call
            self.action_log.add_step(
                module="Supervisor",
                submodule="DecisionMaking",
                prompt=prompt,
                response=response
            )

            decision = json.loads(response)

            # Validate structure
            required_keys = ["reasoning", "action", "parameters"]
            if not all(k in decision for k in required_keys):
                print("Invalid decision format")
                return None

            return decision

        except json.JSONDecodeError as e:
            print(f"Failed to parse decision JSON: {e}")
            return None
        except Exception as e:
            print(f"Decision-making error: {e}")
            return None

    # ========================================================================
    # Action Execution
    # ========================================================================
    def _action_collect_requirements(self) -> str:
        """
        Collect user requirements (no LLM call — pure I/O).
        """
        print("\n📋 Collecting user requirements...")
        self.user_requirements = self.user_comm.get_vehicle_request()
        self.actions_taken.add("collect_requirements")
        return str(self.user_requirements)

    def _action_search_vehicle_models(self) -> str:
        """
        Load the RAG result JSON and populate self.vehicle_models.

        The JSON file has the same shape that the RAG agent will eventually
        produce at runtime:
            { "vehicles": [ { "make": "...", "model": "...", ... }, ... ] }

        When the real RAG agent is wired in, replace this method body with
        the live RAG call and keep everything else unchanged.
        """
        if not os.path.isfile(self.rag_result_json_path):
            return f"Error: RAG result file not found at {self.rag_result_json_path}"

        print(f"\n🔍 Loading vehicle models from {self.rag_result_json_path}...")

        with open(self.rag_result_json_path, "r", encoding="utf-8") as f:
            rag_result = json.load(f)

        self.vehicle_models = rag_result.get("vehicles", [])

        if not self.vehicle_models:
            return "Warning: JSON loaded but 'vehicles' list is empty"

        self.actions_taken.add("search_vehicle_models") # TODO: if RAG will use LLM calls it should be logged to steps
        return f"Loaded {len(self.vehicle_models)} vehicle models from JSON"

    def _action_retrieve_listings(self) -> str:
        """
        Retrieve listings from CSV using the vehicle models loaded by search.
        """
        if not self.vehicle_models:
            return "Error: Cannot retrieve listings without vehicle models"

        print("Retrieving listings from database...")
        vehicles_result = {"vehicles": self.vehicle_models}
        self.listings = retrieve_listings_from_csv(vehicles_result=vehicles_result)

        total = sum(len(r.get("listings", [])) for r in self.listings.get("results", []))
        self.actions_taken.add("retrieve_listings")
        return f"Retrieved {total} listings"

    def _action_process_listings(self) -> str:
        """
        Delegate to FieldAgent.
        Passes self.action_log so that every LLM call the field agent makes is appended to the same log.
        """
        if not self.listings:
            return "Error: Cannot process without listings"

        print("\n🤖 Delegating to autonomous field agent...")

        field_agent = FieldAgent(
            ads_list=self.listings,
            action_log=self.action_log          # shared log
        )
        self.processed_results = field_agent.process_listings()

        stats = self.processed_results.get("stats", {})
        self.actions_taken.add("process_listings")
        return f"Field agent completed: {stats.get('completed_listings', 0)} listings ready"

    def _action_complete_mission(self) -> str:
        """
        Present final results to the user.
        """
        if not self.processed_results:
            return "Error: No results to present"

        print("\nPresenting results to user...")

        meetings_list = self._extract_meetings(self.processed_results)

        self.user_comm.return_vehicle_ads(
            listings_groups=self.processed_results["results"],
            meetings_list=meetings_list,
            action_log=self.action_log.get_steps()
        )

        self.actions_taken.add("complete_mission")
        return "Results presented to user"

    def _execute_action(self, decision: Dict[str, Any]) -> str:
        """Route a decision to the correct action method."""
        action = decision["action"]
        params = decision.get("parameters", {}) # TODO: use if RAG will require params

        action_map = {
            "search_vehicle_models": self._action_search_vehicle_models,
            "retrieve_listings":  self._action_retrieve_listings,
            "process_listings":   self._action_process_listings,
            "complete_mission":   self._action_complete_mission,
        }

        if action in action_map:
            return action_map[action]()
        return f"Unknown action: {action}"

    # ========================================================================
    # Main Autonomous Loop
    # ========================================================================
    def run(self) -> Dict[str, Any]:
        """
        Run the supervisor loop and return the full step log.

        Returns:
            {
                "results": <processed listings>,
                "stats": <summary>,
                "steps": <action_log steps for /api/execute>
            }
        """
        print("\n" + "=" * 60)
        print("🎯 AUTONOMOUS SUPERVISOR: Starting mission")
        print(f"Goal: Find {self.target_listings} complete listings")
        print("=" * 60)

        # Always start by collecting requirements (no LLM call)
        print("\n[Initialization] Collecting user requirements...")
        self._action_collect_requirements()

        # Main decision loop
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1

            print(f"\n{'=' * 60}")
            print(f"[Cycle {self.iteration_count}] Strategic decision-making...")
            print(f"{'=' * 60}")

            # Check if the mission already complete
            if self._evaluate_mission_success():
                print("✅ Mission goal achieved!")
                self._action_complete_mission()
                break

            # REASON
            decision = self._make_strategic_decision()
            if decision is None:
                print("⚠️ Failed to make decision, stopping")
                break

            print(f"\n💭 Reasoning: {decision['reasoning']}")
            print(f"⚡ Action: {decision['action']}")
            print(f"🎯 Expected: {decision.get('expected_outcome', 'N/A')}")
            print(f"📊 Confidence: {decision.get('confidence', 'unknown')}")

            # ACT
            result = self._execute_action(decision)
            print(f"✅ Result: {result}")

            # Check if this was the terminal action
            if decision["action"] == "complete_mission":
                break

        self._print_summary()

        # Return everything the API layer needs
        return {
            "results": self.processed_results.get("results", []) if self.processed_results else [],
            "stats": self.processed_results.get("stats", {}) if self.processed_results else {},
            "steps": self.action_log.get_steps()
        }

    # ========================================================================
    # Summary & Helpers
    # ========================================================================
    def _print_summary(self):
        """
        Print execution summary to console.
        """
        print("\n" + "=" * 60)
        print("📊 MISSION SUMMARY")
        print("=" * 60)
        print(f"Target: {self.target_listings} complete listings")
        print(f"Actions taken: {', '.join(self.actions_taken)}")
        print(f"Total LLM calls logged: {len(self.action_log.get_steps())}")

        if self.processed_results:
            stats = self.processed_results.get("stats", {})
            print(f"Completed listings: {stats.get('completed_listings', 0)}")

        print("\n" + "=" * 60 + "\n")

    def _extract_meetings(self, field_output: Dict[str, Any]) -> list:
        """
        Pull meeting link lists out of the field agent's output.
        """
        meetings_list = []
        for group in field_output.get("results", []):
            for listing in group.get("listings", []):
                meetings_list.append(listing.get("meetings", []))
        return meetings_list
