"""
AgentSupervisor — LangChain ReAct implementation.

Uses LangChain's create_react_agent + AgentExecutor with a prompt-based
ReAct loop (no tool-calling required), compatible with any chat model.

ActionLog contract:
    - self.action_log is the single ActionLog instance for the entire run.
    - Every LLM call made BY this class is logged via a BaseCallbackHandler.
    - When FieldAgent runs, it receives this same ActionLog and appends
      its own steps to it.  No concatenation needed at the end.
"""

from typing import Dict, Any, Optional

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agents.supervisor_agent.user_communication import UserCommunication
from agents.supervisor_agent.tools import make_supervisor_tools
from agents.search_agents.listings_retriever import retrieve_listings_from_csv
from agents.field_agent.field_agent import FieldAgent
from gateways.llm_gateway import LLMGateway
from agents.action_log import ActionLog
from agents.prompts import SUPERVISOR_REACT_PROMPT

from dotenv import load_dotenv
import os
import json
from config import MAX_DECISION_ITERATIONS, NUM_TARGET_LISTINGS, MAX_RECOMMENDED_VEHICLES

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ============================================================================
# ActionLog Callback Handler
# ============================================================================

class SupervisorLogCallback(BaseCallbackHandler):
    """
    Intercepts every LLM call made by the AgentExecutor and appends a step
    to the shared ActionLog with the required schema:
        {module, submodule, prompt, response}
    """

    def __init__(self, action_log: ActionLog):
        super().__init__()
        self._log = action_log
        self._last_prompt: str = ""

    def on_llm_start(self, serialized, prompts: list, **kwargs):
        """Capture the prompt text before the LLM call."""
        self._last_prompt = "\n".join(prompts) if prompts else ""

    def on_llm_end(self, response: LLMResult, **kwargs):
        """After the LLM responds, log prompt + response to ActionLog."""
        try:
            response_text = response.generations[0][0].text
        except (IndexError, AttributeError):
            response_text = str(response)

        self._log.add_step(
            module="Supervisor",
            submodule="DecisionMaking",
            prompt=self._last_prompt,
            response=response_text,
        )


# ============================================================================
# AgentSupervisor
# ============================================================================

class AgentSupervisor:
    """
    Autonomous supervisor that coordinates the car-finding mission.

    Logging contract:
        - self.action_log is the single ActionLog instance for the entire run.
        - Every LLM call made BY this class is logged here via callback.
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
        self.vehicle_models = None
        self.listings = None
        self.processed_results = None

        # Execution tracking
        self.actions_taken: set = set()

        # Build the LangChain ReAct agent
        llm = llm_gateway.client
        tools = make_supervisor_tools(self)

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=SUPERVISOR_REACT_PROMPT,
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iterations,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[SupervisorLogCallback(self.action_log)],
        )

    # ========================================================================
    # Action Methods  (called by LangChain tool wrappers in tools.py)
    # ========================================================================
    def _action_search_vehicle_models(self) -> str:
        """Load vehicle models from the RAG result JSON."""
        if not os.path.isfile(self.rag_result_json_path):
            return f"Error: RAG result file not found at {self.rag_result_json_path}"

        print(f"\n🔍 Loading vehicle models from {self.rag_result_json_path}...")

        with open(self.rag_result_json_path, "r", encoding="utf-8") as f:
            rag_result = json.load(f)

        self.vehicle_models = rag_result.get("vehicles", [])

        if not self.vehicle_models:
            return "Warning: JSON loaded but 'vehicles' list is empty"

        self.actions_taken.add("search_vehicle_models")
        return f"Loaded {len(self.vehicle_models)} vehicle models from JSON"

    def _action_retrieve_listings(self) -> str:
        """Retrieve listings from CSV using the found vehicle models."""
        if not self.vehicle_models:
            return "Error: Cannot retrieve listings without vehicle models"

        print("Retrieving listings from database...")
        vehicles_result = {"vehicles": self.vehicle_models}
        self.listings = retrieve_listings_from_csv(vehicles_result=vehicles_result)

        total = sum(len(r.get("listings", [])) for r in self.listings.get("results", []))
        self.actions_taken.add("retrieve_listings")
        return f"Retrieved {total} listings"

    def _action_process_listings(self) -> str:
        """Delegate to FieldAgent for listing completion and scheduling."""
        if not self.listings:
            return "Error: Cannot process without listings"

        print("\n🤖 Delegating to autonomous field agent...")

        field_agent = FieldAgent(
            ads_list=self.listings,
            action_log=self.action_log,  # shared log
        )
        self.processed_results = field_agent.process_listings()

        stats = self.processed_results.get("stats", {})
        self.actions_taken.add("process_listings")
        return f"Field agent completed: {stats.get('completed_listings', 0)} listings ready"

    def _action_complete_mission(self) -> str:
        """Present final results to the user."""
        if not self.processed_results:
            return "Error: No results to present"

        print("\nPresenting results to user...")

        meetings_list = self._extract_meetings(self.processed_results)
        self.user_comm.return_vehicle_ads(
            listings_groups=self.processed_results["results"],
            meetings_list=meetings_list,
            action_log=self.action_log.get_steps(),
        )

        self.actions_taken.add("complete_mission")
        return "Results presented to user"

    # ========================================================================
    # Main Autonomous Loop (LangChain ReAct)
    # ========================================================================
    def run(self) -> Dict[str, Any]:
        """
        Run the LangChain ReAct supervisor loop and return the full step log.

        Returns:
            {
                "results": <processed listings>,
                "stats":   <summary>,
                "steps":   <action_log steps for /api/execute>
            }
        """
        print("\n" + "=" * 60)
        print("🎯 AUTONOMOUS SUPERVISOR: Starting mission (LangChain ReAct)")
        print(f"Goal: Find {self.target_listings} complete listings")
        print("=" * 60)

        # Always collect requirements first (no LLM call)
        print("\n[Initialization] Collecting user requirements...")
        self.user_requirements = self.user_comm.get_vehicle_request()
        self.actions_taken.add("collect_requirements")

        # Build the input prompt for the ReAct agent
        agent_input = (
            f"Find {self.target_listings} complete vehicle listings and schedule meetings.\n\n"
            f"User requirements have been collected:\n"
            f"  Max price: ${self.user_requirements.get('max_price', 'N/A')}\n"
            f"  Min year: {self.user_requirements.get('year_min', 'N/A')}\n"
            f"  Description: {self.user_requirements.get('description', 'N/A')}\n\n"
            f"Execute the tools in order: search_vehicle_models → retrieve_listings → "
            f"process_listings → complete_mission."
        )

        self._executor.invoke({"input": agent_input})

        self._print_summary()

        return {
            "results": self.processed_results.get("results", []) if self.processed_results else [],
            "stats": self.processed_results.get("stats", {}) if self.processed_results else {},
            "steps": self.action_log.get_steps(),
        }

    # ========================================================================
    # Summary & Helpers
    # ========================================================================
    def _print_summary(self):
        """Print execution summary to console."""
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
        """Pull meeting link lists out of the field agent's output."""
        meetings_list = []
        for group in field_output.get("results", []):
            for listing in group.get("listings", []):
                meetings_list.append(listing.get("meetings", []))
        return meetings_list
