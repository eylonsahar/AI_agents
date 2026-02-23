"""
LangChain tools for the FieldAgent.

Each tool is created via a factory function that closes over the live
FieldAgent instance, so the tools can read/mutate agent state and write
to the shared ActionLog without any global variables.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from langchain_core.tools import tool

if TYPE_CHECKING:
    from agents.field_agent.field_agent import FieldAgent


def make_field_agent_tools(agent: "FieldAgent") -> list:
    """
    Build and return the list of LangChain Tool objects bound to *agent*.

    Args:
        agent: A live FieldAgent instance whose methods the tools will call.

    Returns:
        A list of langchain Tool objects ready to pass to create_react_agent.
    """

    @tool
    def fill_missing_data(tool_input: str) -> str:
        """Contact the seller to obtain missing fields for a vehicle listing.

        Use this tool whenever a listing has one or more missing or empty
        critical fields (price, year, manufacturer, model, mileage, accident,
        condition, paint_color, state).  Request ALL missing fields in a single
        call.

        Input: JSON with "listing_id" (string) and "fields_to_request" (list of field names).
        Example: {"listing_id": "1001", "fields_to_request": ["mileage", "accident", "condition"]}

        Returns:
            A short status message confirming how many fields were filled.
        """
        try:
            data = json.loads(tool_input)
            listing_id = data["listing_id"]
            fields_to_request = data["fields_to_request"]
        except (json.JSONDecodeError, KeyError) as e:
            return f"Error parsing input: {e}. Expected JSON with 'listing_id' and 'fields_to_request'."
        return agent._tool_fill_missing_data(
            listing_id=listing_id,
            fields_to_request=fields_to_request,
        )

    @tool
    def schedule_meeting(tool_input: str) -> str:
        """Generate Google Calendar meeting links for a fully-completed listing.

        Only call this tool after ALL critical fields of a listing are filled.
        The tool contacts the mock seller for available time slots and attaches
        calendar links to the listing record.

        Input: JSON with "listing_id" (string).
        Example: {"listing_id": "1001"}

        Returns:
            A short status message confirming how many slots were scheduled.
        """
        try:
            data = json.loads(tool_input)
            listing_id = data["listing_id"]
        except (json.JSONDecodeError, KeyError) as e:
            return f"Error parsing input: {e}. Expected JSON with 'listing_id'."
        return agent._tool_schedule_meeting(listing_id=listing_id)

    @tool
    def complete_processing(tool_input: str = "") -> str:
        """Signal that ALL listings have been fully processed and have meetings scheduled.

        Call this tool only when every single listing has meetings scheduled.
        This terminates the agent loop.

        Input: Empty string or any value (ignored).

        Returns:
            A confirmation message that processing is complete.
        """
        return agent._tool_complete_processing()

    return [fill_missing_data, schedule_meeting, complete_processing]
