"""
FieldAgent — LangChain ReAct implementation.

Uses LangChain's create_react_agent + AgentExecutor with a prompt-based
ReAct loop (no tool-calling required), compatible with any chat model.

ActionLog contract:
    Receives an ActionLog from the Supervisor.  Every LLM call this agent
    makes — decision calls AND MockSeller calls — is appended to that same log.
    A BaseCallbackHandler intercepts each LLM call and appends a step with the
    required schema {module, submodule, prompt, response}.
"""

import os
import json
import urllib.parse
from typing import Dict, List, Any, Tuple, Optional

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agents.field_agent.mock_seller import MockSeller
from agents.field_agent.tools import make_field_agent_tools
from agents.action_log import ActionLog
from agents.prompts import FIELD_AGENT_REACT_PROMPT
from gateways.llm_gateway import LLMGateway

from datetime import datetime, timedelta
from dotenv import load_dotenv
from config import (NUM_AVAILABLE_DATES, MEETING_DURATION, MEETING_TIMEFRAME,
                    GUARANTEED_MISSING_FIELDS, CRITICAL_FIELDS, MAX_DECISION_ITERATIONS)

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ============================================================================
# ActionLog Callback Handler
# ============================================================================

class FieldAgentLogCallback(BaseCallbackHandler):
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
            module="FieldAgent",
            submodule="DecisionMaking",
            prompt=self._last_prompt,
            response=response_text,
        )


# ============================================================================
# FieldAgent
# ============================================================================

class FieldAgent:
    """
    Autonomous agent that contacts sellers to complete listings and schedule meetings.

    Logging contract:
        Receives an ActionLog from the Supervisor.  Every LLM call this agent
        makes — decision calls AND MockSeller calls — is appended to that same log.
        The Supervisor does not need to do any concatenation.
    """

    def __init__(self, ads_list: dict, action_log: ActionLog, max_iterations: int = MAX_DECISION_ITERATIONS):
        """
        Args:
            ads_list:       Vehicle listings to process (from listings retriever or RAG).
            action_log:     Shared ActionLog instance (owned by Supervisor).
            max_iterations: Safety cap on AgentExecutor iterations.
        """
        self.ads_list = ads_list
        self.action_log = action_log
        self.max_iterations = max_iterations

        # Track state
        self.processed_listings: Dict[str, str] = {}
        self.completed_listings: List[str] = []
        self.iteration_count = 0

        # Build the LangChain ReAct agent
        llm = LLMGateway(api_key=OPENAI_API_KEY).client
        tools = make_field_agent_tools(self)

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=FIELD_AGENT_REACT_PROMPT,
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iterations,
            verbose=True,
            handle_parsing_errors=False,  # Disable error handling to see the actual issue
            callbacks=[FieldAgentLogCallback(action_log)],
        )

    # ========================================================================
    # State Management
    # ========================================================================
    def _get_current_state(self) -> str:
        """Generate the state string passed to the agent as input."""
        state_parts = []

        total_listings = sum(
            len(result.get("listings", []))
            for result in self.ads_list.get("results", [])
        )

        state_parts.append(f"TOTAL LISTINGS: {total_listings}")
        state_parts.append(f"COMPLETED WITH MEETINGS: {len(self.completed_listings)}")
        state_parts.append("")
        state_parts.append("LISTING STATUS (use these exact listing_id values):")
        state_parts.append("")

        for result in self.ads_list.get("results", []):
            vehicle_info = result.get("vehicle", {})
            vehicle_name = f"{vehicle_info.get('make')} {vehicle_info.get('model')}"
            state_parts.append(f"{vehicle_name}:")

            for listing in result.get("listings", []):
                listing_id = str(listing.get("id"))
                has_meetings = listing_id in self.completed_listings
                missing_fields = self._identify_missing_fields(listing)

                if has_meetings:
                    status = "COMPLETE - Has meetings scheduled"
                elif not missing_fields:
                    status = "READY FOR SCHEDULING - All fields filled, needs meetings"
                else:
                    status = f"MISSING FIELDS: {', '.join(missing_fields)}"

                state_parts.append(f"  listing_id: {listing_id}")
                state_parts.append(f"  Status: {status}")
                state_parts.append("")

        # Next steps guidance
        ready_for_scheduling = []
        needs_data = []

        for result in self.ads_list.get("results", []):
            for listing in result.get("listings", []):
                listing_id = str(listing.get("id"))
                if listing_id in self.completed_listings:
                    continue
                missing = self._identify_missing_fields(listing)
                if not missing:
                    ready_for_scheduling.append(listing_id)
                else:
                    needs_data.append(f"{listing_id} (needs: {', '.join(missing)})")

        state_parts.append("NEXT STEPS:")
        if ready_for_scheduling:
            state_parts.append(f"  Ready to schedule meetings: {', '.join(ready_for_scheduling)}")
        if needs_data:
            state_parts.append(f"  Needs data filled: {' | '.join(needs_data)}")

        if len(self.completed_listings) == total_listings:
            state_parts.append("  ALL LISTINGS COMPLETE - Call complete_processing now")
        else:
            remaining = total_listings - len(self.completed_listings)
            state_parts.append(f"  {remaining} listings still need meetings scheduled")

        return "\n".join(state_parts)

    def _identify_missing_fields(self, listing: dict) -> List[str]:
        """Return the list of critical fields that are missing or empty."""
        missing = []
        all_critical = list(GUARANTEED_MISSING_FIELDS) + [
            f for f in CRITICAL_FIELDS if f not in GUARANTEED_MISSING_FIELDS
        ]
        for field in all_critical:
            val = listing.get(field)
            if val is None or val == "" or val == "Not Provided":
                missing.append(field)
        return list(set(missing))

    def _find_listing_by_id(self, listing_id: str) -> Tuple[Optional[dict], Optional[dict]]:
        """Return (listing, vehicle_context) for the given ID, or (None, None) if not found."""
        for result in self.ads_list.get("results", []):
            vehicle_context = result.get("vehicle", {})
            for listing in result.get("listings", []):
                if str(listing.get("id")) == str(listing_id):
                    return listing, vehicle_context
        return None, None

    # ========================================================================
    # Tools  (called by LangChain tool wrappers in tools.py)
    # ========================================================================
    def _tool_fill_missing_data(self, listing_id: str, fields_to_request: List[str]) -> str:
        """Contact the mock seller to fill missing fields."""
        listing, vehicle_context = self._find_listing_by_id(listing_id)
        if not listing:
            return f"Error: Listing {listing_id} not found"

        query = f"Provide these fields: {', '.join(fields_to_request)} for listing ID {listing_id}"
        seller = MockSeller(query_from_field_agent=query)
        raw_response, _ = seller.get_missing_data()

        # Log the MockSeller LLM call
        self.action_log.add_step(
            module="FieldAgent",
            submodule="MockSeller/GetData",
            prompt=f"{seller.info_system_prompt}\n\nAgent Query: {query}",
            response=raw_response,
        )

        seller_dict = self._parse_seller_response(raw_response)
        filled_count = 0
        for field in fields_to_request:
            if field in seller_dict and seller_dict[field]:
                listing[field] = seller_dict[field]
                filled_count += 1

        self.processed_listings[listing_id] = "data_filled"
        return f"Filled {filled_count}/{len(fields_to_request)} fields for listing {listing_id}"

    def _tool_schedule_meeting(self, listing_id: str) -> str:
        """Generate calendar links for a listing."""
        listing, vehicle_context = self._find_listing_by_id(listing_id)
        if not listing:
            return f"Error: Listing {listing_id} not found"

        meetings = self._get_meeting_slots(listing)
        listing["meetings"] = meetings
        self.completed_listings.append(listing_id)
        self.processed_listings[listing_id] = "completed"

        return f"Scheduled {len(meetings)} meeting slots for listing {listing_id}"

    def _tool_complete_processing(self) -> str:
        """Mark all work as complete."""
        return "Processing marked as complete"

    # ========================================================================
    # Main Agent Loop (LangChain ReAct)
    # ========================================================================
    def process_listings(self) -> Dict[str, Any]:
        """
        Run the LangChain ReAct loop to process all listings.

        Returns:
            {
                "results": <listings with filled data and meetings>,
                "stats":   <summary counts>
            }
            Note: steps are already in self.action_log (shared with Supervisor).
        """
        state = self._get_current_state()

        self._executor.invoke({
            "input": (
                f"Process all vehicle listings below. For each listing: fill any missing fields, "
                f"then schedule a meeting. Call complete_processing when ALL listings are done.\n\n"
                f"{state}"
            )
        })

        return {
            "results": self.ads_list.get("results", []),
            "stats": {
                "total_decisions": self.max_iterations,
                "completed_listings": len(self.completed_listings),
                "processed_listings": len(self.processed_listings),
            },
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================
    def _parse_seller_response(self, raw_response: str) -> dict:
        """Parse 'key = value' lines into a dict."""
        data = {}
        for line in raw_response.strip().split('\n'):
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip().lower()] = value.strip()
        return data

    def _get_meeting_slots(self, listing: dict) -> List[Dict[str, str]]:
        """Ask the mock seller for available slots, then build calendar links."""
        today = datetime.now()
        two_weeks_later = today + timedelta(days=MEETING_TIMEFRAME)

        query = (
            f"Generate exactly {NUM_AVAILABLE_DATES} available slots.\n"
            f"Date Window: {today.strftime('%Y-%m-%d')} to {two_weeks_later.strftime('%Y-%m-%d')}.\n"
            "Days: Sunday to Friday ONLY.\n"
            "Hours: 10:00 to 19:00.\n"
            "Minutes: :00, :15, :30, or :45.\n"
            "Return ONLY the list of slots, one per line."
        )

        seller = MockSeller(query_from_field_agent=query)
        available_slots, _ = seller.get_available_dates()

        # Log the scheduling LLM call
        self.action_log.add_step(
            module="FieldAgent",
            submodule="MockSeller/Scheduling",
            prompt=seller.sched_system_prompt + query,
            response="\n".join(available_slots),
        )

        slot_links = []
        for slot in available_slots:
            url = self._create_calendar_link(slot, listing)
            slot_links.append({"slot": slot, "url": url})

        return slot_links

    def _create_calendar_link(self, slot: str, listing: dict) -> str:
        """Generate a Google Calendar event URL for a single slot."""
        start_time = datetime.strptime(slot, "%Y-%m-%d %H:%M")
        end_time = start_time + timedelta(minutes=MEETING_DURATION)

        fmt = "%Y%m%dT%H%M%S"
        dates_param = f"{start_time.strftime(fmt)}/{end_time.strftime(fmt)}"

        params = {
            "action": "TEMPLATE",
            "text": f"View Car: {listing.get('manufacturer', '')} {listing.get('model', '')}",
            "dates": dates_param,
            "details": f"Car ID: {listing.get('id')}",
            "sf": "true",
        }
        return f"https://www.google.com/calendar/render?{urllib.parse.urlencode(params)}"

    def print_action_log(self):
        """Print the shared action log."""
        self.action_log.print_steps()
