"""
LangChain tools for the AgentSupervisor.

Each tool is created via a factory function that closes over the live
AgentSupervisor instance, allowing the tools to mutate supervisor state
and write to the shared ActionLog without any global variables.
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
    def search_vehicle_models() -> str:
        """Search for vehicle models that match the user's requirements.

        Use this as the first action after user requirements have been collected.
        Loads vehicle model candidates from the RAG result source and stores them
        in the supervisor for use in the next step.

        Returns:
            A status message stating how many vehicle models were found.
        """
        return supervisor._action_search_vehicle_models()

    @tool
    def retrieve_listings() -> str:
        """Retrieve for-sale vehicle listings from the database.

        Use this after vehicle models have been found by search_vehicle_models.
        Fetches actual listings from the CSV database for each vehicle model.

        Returns:
            A status message stating how many listings were retrieved.
        """
        return supervisor._action_retrieve_listings()

    @tool
    def process_listings() -> str:
        """Delegate listing completion and meeting scheduling to the Field Agent.

        Use this after listings have been retrieved. The Field Agent will:
        - Fill in any missing data fields by contacting mock sellers
        - Schedule meetings for each completed listing

        Returns:
            A status message with how many listings the field agent completed.
        """
        return supervisor._action_process_listings()

    @tool
    def complete_mission() -> str:
        """Present the final results to the user and end the mission.

        Use this only when all listings are fully processed and have meetings
        scheduled. This is the terminal action that ends the supervisor loop.

        Returns:
            A confirmation that results were presented to the user.
        """
        return supervisor._action_complete_mission()

    return [search_vehicle_models, retrieve_listings, process_listings, complete_mission]
