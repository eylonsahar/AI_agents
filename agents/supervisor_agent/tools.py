"""
LangChain tools for the AgentSupervisor.

Each tool is created via a factory function that closes over the live
AgentSupervisor instance, allowing the tools to mutate supervisor state
and write to the shared ActionLog without any global variables.
"""

from __future__ import annotations

import json
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
    def search_vehicles(tool_input: str) -> str:
        """Search for vehicle models and listings matching the user's query.

        Use this as the first action. Runs the full search pipeline:
        1. Finds matching vehicle models using RAG
        2. Retrieves listings from the database
        3. Scores and ranks the listings

        Input: JSON with "query" (string) - the user's search query.
        Example: {"query": "reliable SUV for family under $30000"}

        Returns:
            A status message with how many vehicle models and listings were found.
        """
        if supervisor._mission_complete:
            return "Mission already complete. Do not continue searching."
        
        try:
            data = json.loads(tool_input)
            query = data.get("query", tool_input)
        except (json.JSONDecodeError, AttributeError):
            # If not JSON, treat the input as the query directly
            query = tool_input.strip()
        
        return supervisor._action_search_vehicles(query)

    @tool
    def process_listings(tool_input: str = "") -> str:
        """Delegate listing completion and meeting scheduling to the Field Agent.

        Use this after search_vehicles has found listings. The Field Agent will:
        - Fill in any missing data fields by contacting mock sellers
        - Schedule meetings for each completed listing

        Input: Empty string or any value (ignored).

        Returns:
            A status message with how many listings the field agent completed.
        """
        if supervisor._mission_complete:
            return "Mission already complete. Do not continue processing."
        
        return supervisor._action_process_listings()

    @tool
    def complete_mission(tool_input: str = "") -> str:
        """Present the final results to the user and end the mission.

        Use this only when all listings are fully processed and have meetings
        scheduled. This is the terminal action that ends the supervisor loop.

        Input: Empty string or any value (ignored).

        Returns:
            A confirmation that results were presented to the user.
        """
        return supervisor._action_complete_mission()

    return [search_vehicles, process_listings, complete_mission]
