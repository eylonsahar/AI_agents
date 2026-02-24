"""
LangChain tools for the AgentSupervisor.

Each tool is created via a factory function that closes over the live
AgentSupervisor instance, allowing the tools to mutate supervisor state
and write to the shared ActionLog without any global variables.

IMPORTANT: All tools are idempotent — calling a tool that has already
run successfully returns an instruction to proceed to the next step
instead of re-executing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from langchain_core.tools import tool

if TYPE_CHECKING:
    from agents.supervisor_agent.supervisor_agent import AgentSupervisor


def make_supervisor_tools(supervisor: "AgentSupervisor") -> list:
    """
    Build and return the list of LangChain Tool objects bound to *supervisor*.

    Args:
        supervisor: A live AgentSupervisor instance whose methods the tools call.

    Returns:
        A list of langchain Tool objects ready to pass to create_react_agent.
    """

    @tool
    def search_vehicle_models(tool_input: str = "") -> str:
        """Search for vehicle models that match the user's requirements.

        Use this EXACTLY ONCE as the first action.
        If this tool has already been called successfully, do NOT call it again.

        Returns:
            A status message stating how many vehicle models were found.
        """
        # Idempotency guard: already found models → tell agent to move on
        if "search_vehicle_models" in supervisor.actions_taken and supervisor.vehicle_models:
            return (
                f"Already found {len(supervisor.vehicle_models)} vehicle models "
                f"({', '.join(v.get('make','') + ' ' + v.get('model','') for v in supervisor.vehicle_models)}). "
                "Proceed to retrieve_listings NOW."
            )
        return supervisor._action_search_vehicle_models()

    @tool
    def retrieve_listings(tool_input: str = "") -> str:
        """Retrieve for-sale vehicle listings from the database.

        Use this EXACTLY ONCE after search_vehicle_models has succeeded.
        Do NOT call this before search_vehicle_models.
        Do NOT call this more than once.

        Returns:
            A status message stating how many listings were retrieved.
        """
        if "retrieve_listings" in supervisor.actions_taken and supervisor.listings:
            total = sum(len(r.get("listings", [])) for r in supervisor.listings.get("results", []))
            return (
                f"Already retrieved {total} listings. "
                "Proceed to process_listings NOW."
            )
        return supervisor._action_retrieve_listings()

    @tool
    def process_listings(tool_input: str = "") -> str:
        """Delegate listing completion and meeting scheduling to the Field Agent.

        Use this EXACTLY ONCE after listings have been retrieved.
        Do NOT call this before retrieve_listings.
        Do NOT call this more than once.

        Returns:
            A status message with how many listings the field agent completed.
        """
        if "process_listings" in supervisor.actions_taken and supervisor.processed_results:
            stats = supervisor.processed_results.get("stats", {})
            return (
                f"Already processed {stats.get('completed_listings', 0)} listings. "
                "Proceed to complete_mission NOW."
            )
        return supervisor._action_process_listings()

    @tool
    def complete_mission(tool_input: str = "") -> str:
        """Present the final results to the user and end the mission.

        Use this EXACTLY ONCE after process_listings has completed.
        This is the terminal action — after calling it, output Final Answer.

        Returns:
            A confirmation that results were presented to the user.
        """
        if "complete_mission" in supervisor.actions_taken:
            return "Mission already complete. Output Final Answer now."
        return supervisor._action_complete_mission()

    return [search_vehicle_models, retrieve_listings, process_listings, complete_mission]
