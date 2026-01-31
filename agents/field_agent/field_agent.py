import os
import json
import urllib.parse
from agents.field_agent.mock_seller import MockSeller
from gateways.llm_gateway import LLMGateway
from agents.action_log import ActionLog
from datetime import datetime, timedelta
from agents.prompts import FIELD_AGENT_DECISION_PROMPT
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from config import (NUM_AVAILABLE_DATES, MEETING_DURATION, MEETING_TIMEFRAME,
                    GUARANTEED_MISSING_FIELDS, CRITICAL_FIELDS, MAX_DECISION_ITERATIONS)

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


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
            ads_list:      Vehicle listings to process (from listings retriever or RAG).
            action_log:    Shared ActionLog instance (owned by Supervisor).
            max_iterations: Safety cap on decision cycles.
        """
        self.ads_list = ads_list
        self.llm = LLMGateway(api_key=OPENAI_API_KEY)
        self.action_log = action_log
        self.max_iterations = max_iterations

        # Track state
        self.processed_listings = {}   # listing_id -> status string
        self.completed_listings = []   # listing_ids that have meetings scheduled
        self.iteration_count = 0

    # ========================================================================
    # State Management
    # ========================================================================
    def _get_current_state(self) -> str:
        """
        Generate the state string injected into the decision prompt each cycle.
        """
        state_parts = []

        # Count the number of listings that are being processed in the field agent
        total_listings = sum(
            len(result.get("listings", []))
            for result in self.ads_list.get("results", [])
        )

        state_parts.append(f"TOTAL LISTINGS: {total_listings}")
        state_parts.append(f"PROCESSED: {len(self.processed_listings)}")
        state_parts.append(f"COMPLETED WITH MEETINGS: {len(self.completed_listings)}")
        state_parts.append(f"ITERATION: {self.iteration_count}/{self.max_iterations}")
        state_parts.append("")

        # Per-listing status
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

                state_parts.append(f" listing_id: {listing_id}")
                state_parts.append(f" Status: {status}")
                state_parts.append("")

        # Guidance section - what is the action required for each incomplete listing
        state_parts.append("NEXT STEPS:")

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

        if ready_for_scheduling:
            state_parts.append(f"Ready to schedule meetings: {', '.join(ready_for_scheduling)}")
        if needs_data:
            state_parts.append(f"Needs data filled: {' | '.join(needs_data)}")

        if len(self.completed_listings) == total_listings:
            state_parts.append("ALL LISTINGS COMPLETE - Ready to call complete_processing")
        else:
            remaining = total_listings - len(self.completed_listings)
            state_parts.append(f"{remaining} listings still need meetings scheduled")

        return "\n".join(state_parts)

    def _identify_missing_fields(self, listing: dict) -> List[str]:
        """
        Check what fields are missing out of the fields that have attributes in the data
        and the extra attributes we required.
        Return the list of critical fields that are missing or empty.
        """
        missing = []

        all_critical = list(GUARANTEED_MISSING_FIELDS) + [f for f in CRITICAL_FIELDS if f not in GUARANTEED_MISSING_FIELDS]
        for field in all_critical:
            val = listing.get(field)
            if val is None or val == "" or val == "Not Provided":
                missing.append(field)
        return list(set(missing))

    def _find_listing_by_id(self, listing_id: str) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Return (listing, vehicle_context) for the given ID, or (None, None) if ID not found.
        """
        for result in self.ads_list.get("results", []):
            vehicle_context = result.get("vehicle", {})
            for listing in result.get("listings", []):
                if str(listing.get("id")) == str(listing_id):
                    return listing, vehicle_context
        return None, None

    # ========================================================================
    # Decision Making
    # ========================================================================
    def _make_decision(self) -> Optional[Dict[str, Any]]:
        """
        Call LLM to decide the next action.
        Logs the call to self.action_log.
        """
        state = self._get_current_state()
        prompt = FIELD_AGENT_DECISION_PROMPT.format(state=state)

        try:
            response, _ = self.llm.call_llm(prompt=prompt)

            # Log this LLM call
            self.action_log.add_step(
                module="FieldAgent",
                submodule="DecisionMaking",
                prompt=prompt,
                response=response
            )

            decision = json.loads(response)

            required_keys = ["reasoning", "action", "parameters"]
            if not all(k in decision for k in required_keys):
                print(f"Invalid decision format: {decision}")
                return None

            return decision

        except json.JSONDecodeError as e:
            print(f"Failed to parse decision JSON: {e}")
            return None
        except Exception as e:
            print(f"Decision-making error: {e}")
            return None

    # ========================================================================
    # Tools
    # ========================================================================
    def _tool_fill_missing_data(self, listing_id: str, fields_to_request: List[str]) -> str:
        """
        Contact the mock seller to fill missing fields.
        """
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
            response=raw_response
        )

        # Parse and update listing
        seller_dict = self._parse_seller_response(raw_response)
        filled_count = 0
        for field in fields_to_request:
            if field in seller_dict and seller_dict[field]:
                listing[field] = seller_dict[field]
                filled_count += 1

        self.processed_listings[listing_id] = "data_filled"
        return f"Filled {filled_count}/{len(fields_to_request)} fields for listing {listing_id}"

    def _tool_schedule_meeting(self, listing_id: str) -> str:
        """
        Generate calendar links for a listing.
        """
        listing, vehicle_context = self._find_listing_by_id(listing_id)
        if not listing:
            return f"Error: Listing {listing_id} not found"

        meetings = self._get_meeting_slots(listing)

        listing["meetings"] = meetings
        self.completed_listings.append(listing_id)
        self.processed_listings[listing_id] = "completed"

        return f"Scheduled {len(meetings)} meeting slots for listing {listing_id}"

    def _tool_complete_processing(self) -> str:
        """
        Mark all work as complete.
        """
        return "Processing marked as complete"

    # ========================================================================
    # Action Execution
    # ========================================================================
    def _execute_action(self, decision: Dict[str, Any]) -> str:
        """
        Route a decision to the correct tool.
        """
        action = decision["action"]     # Chosen action
        params = decision["parameters"] # Parameters needed for that function

        if action == "fill_missing_data":
            return self._tool_fill_missing_data(
                listing_id=params["listing_id"],
                fields_to_request=params.get("fields_to_request", GUARANTEED_MISSING_FIELDS)
            )
        elif action == "schedule_meeting":
            return self._tool_schedule_meeting(listing_id=params["listing_id"])
        elif action == "complete_processing":
            return self._tool_complete_processing()
        else:
            return f"Unknown action: {action}"

    # ========================================================================
    # Main Agent Loop (ReAct)
    # ========================================================================
    def process_listings(self) -> Dict[str, Any]:
        """
        ReAct loop: Observe → Reason → Act → Reflect → repeat.

        Returns:
            {
                "results": <listings with filled data and meetings>,
                "stats":   <summary counts>
            }
            Note: steps are already in self.action_log (shared with Supervisor).
        """
        print("\n" + "=" * 60)
        print("🤖 AUTONOMOUS FIELD AGENT: Starting")
        print("=" * 60)

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n[Cycle {self.iteration_count}] Reasoning about next action...")

            # REASON
            decision = self._make_decision()
            if decision is None:
                print("Failed to make decision, stopping")
                break

            print(f"Reasoning: {decision['reasoning']}")
            print(f"Action: {decision['action']}")

            # Terminal action
            if decision["action"] == "complete_processing":
                print("Agent decided processing is complete")
                break

            # ACT
            result = self._execute_action(decision)
            print(f"Result: {result}")

        print("\n" + "=" * 60)
        print("✨ AUTONOMOUS FIELD AGENT: Complete")
        print(f"📊 Decisions made: {self.iteration_count}")
        print(f"✅ Completed: {len(self.completed_listings)}")
        print("=" * 60 + "\n")

        return {
            "results": self.ads_list.get("results", []),
            "stats": {
                "total_decisions": self.iteration_count,
                "completed_listings": len(self.completed_listings),
                "processed_listings": len(self.processed_listings)
            }
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================
    def _parse_seller_response(self, raw_response: str) -> dict:
        """
        Parse 'key = value' lines into a dict.
        """
        data = {}
        for line in raw_response.strip().split('\n'):
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip().lower()] = value.strip()
        return data

    def _get_meeting_slots(self, listing: dict) -> List[Dict[str, str]]:
        """
        Ask the mock seller for available slots, then build calendar links.
        """
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
            response="\n".join(available_slots)
        )

        slot_links = []
        for slot in available_slots:
            url = self._create_calendar_link(slot, listing)
            slot_links.append({"slot": slot, "url": url})

        return slot_links

    def _create_calendar_link(self, slot: str, listing: dict) -> str:
        """
        Generate a Google Calendar event URL for a single slot.
        """
        start_time = datetime.strptime(slot, "%Y-%m-%d %H:%M")
        end_time = start_time + timedelta(minutes=MEETING_DURATION)

        fmt = "%Y%m%dT%H%M%S"
        dates_param = f"{start_time.strftime(fmt)}/{end_time.strftime(fmt)}"

        params = {
            "action": "TEMPLATE",
            "text": f"View Car: {listing.get('manufacturer', '')} {listing.get('model', '')}",
            "dates": dates_param,
            "details": f"Car ID: {listing.get('id')}",
            "sf": "true"
        }
        return f"https://www.google.com/calendar/render?{urllib.parse.urlencode(params)}"

    def print_action_log(self):
        """
        Print the shared action log.
        """
        self.action_log.print_steps()
