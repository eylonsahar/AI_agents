"""
LangChain tools for the FieldAgent.

Each tool is created via a factory function that closes over the live
FieldAgent instance, so the tools can read/mutate agent state and write
to the shared ActionLog without any global variables.

NOTE: All tools accept a SINGLE STRING input (JSON or plain ID) and parse
it internally.  This avoids LangChain ReAct's multi-parameter parsing bug
where the entire JSON block gets mapped to the first argument.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from langchain_core.tools import tool

if TYPE_CHECKING:
    from agents.field_agent.field_agent import FieldAgent


def _parse_json_or_plain(text: str) -> dict:
    """
    Parse tool input that may be:
      - A valid JSON object  {"listing_id": "x", "fields_to_request": [...]}
      - A plain string       "7316766563"
    Returns a dict in all cases.
    """
    text = text.strip()
    # Try JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        # e.g. bare JSON string "7316766563"
        return {"listing_id": str(parsed)}
    except json.JSONDecodeError:
        pass
    # Try to extract listing_id with a regex
    m = re.search(r'\d{5,}', text)
    if m:
        return {"listing_id": m.group(0)}
    return {"listing_id": text}


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
        condition, paint_color, state). Request ALL missing fields in one call.

        Pass a JSON string with exactly these two keys:
          {"listing_id": "<id>", "fields_to_request": ["field1", "field2"]}

        Example:
          {"listing_id": "7316766563", "fields_to_request": ["mileage", "accident"]}

        Returns:
            A short status message confirming how many fields were filled.
        """
        data = _parse_json_or_plain(tool_input)
        listing_id = str(data.get("listing_id", "")).strip()
        fields_to_request = data.get("fields_to_request", [])

        if not listing_id:
            return "Error: listing_id is required."
        if not fields_to_request:
            return "Error: fields_to_request must be a non-empty list."
        if isinstance(fields_to_request, str):
            fields_to_request = [f.strip() for f in fields_to_request.split(",")]

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

        Pass the listing ID as a plain string or JSON:
          "7316766563"
          {"listing_id": "7316766563"}

        Returns:
            A short status message confirming how many slots were scheduled.
        """
        data = _parse_json_or_plain(tool_input)
        listing_id = str(data.get("listing_id", tool_input)).strip()
        return agent._tool_schedule_meeting(listing_id=listing_id)

    @tool
    def complete_processing(tool_input: str = "") -> str:
        """Signal that ALL listings have been fully processed and have meetings scheduled.

        Call this tool only when every single listing has meetings scheduled.
        This terminates the agent loop. Pass an empty string or any value.

        Returns:
            A confirmation message that processing is complete.
        """
        return agent._tool_complete_processing()

    return [fill_missing_data, schedule_meeting, complete_processing]
