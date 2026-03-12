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
from agents.search_agent.listings_retriever import ListingsRetriever
from agents.search_agent.vehicle_model_retriever import VehicleModelRetriever
from agents.search_agent.rag_retrieval import get_pinecone_index
from agents.utils.contracts import VehicleModelsResult
from agents.field_agent.field_agent import FieldAgent
from gateways.llm_gateway import LLMGateway
from gateways.embedding_gateway import EmbeddingGateway
from agents.action_log import ActionLog
from agents.prompts import SUPERVISOR_REACT_PROMPT

from dotenv import load_dotenv
import os
import json
from config import MAX_DECISION_ITERATIONS, NUM_TARGET_LISTINGS

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def _split_thought(text: str):
    """
    Split a ReAct LLM response into (thought_text, action_text).

    The Thought: line(s) come first; Action:/Final Answer: comes after.
    Returns (thought_str, rest_str) — either may be empty string.
    """
    import re
    # Find the first occurrence of Action: or Final Answer:
    split_pattern = re.compile(r'(?m)^(Action:|Final Answer:)', re.MULTILINE)
    m = split_pattern.search(text)
    if m:
        thought = text[:m.start()].strip()
        action  = text[m.start():].strip()
    else:
        thought = text.strip()
        action  = ""

    # Strip leading "Thought:" label for cleaner display
    thought = re.sub(r'^Thought:\s*', '', thought, flags=re.IGNORECASE).strip()
    return thought, action


# ============================================================================
# ActionLog Callback Handler
# ============================================================================

class SupervisorLogCallback(BaseCallbackHandler):
    """
    Logs each Supervisor ReAct step to the ActionLog.

    Uses on_agent_action (fires per tool call) and on_agent_finish (fires
    on Final Answer) — both are guaranteed to fire at the AgentExecutor level
    and contain the full Thought text in their .log field.
    """

    def __init__(self, action_log: ActionLog):
        super().__init__()
        self._log = action_log

    def on_agent_action(self, action, **kwargs):
        """Fired before each tool call — log only the Thought."""
        log_text = getattr(action, "log", "") or ""
        thought, _ = _split_thought(log_text)
        if thought:
            self._log.add_step(
                module="Supervisor",
                submodule="Thought",
                prompt="",
                response=thought,
            )

    def on_agent_finish(self, finish, **kwargs):
        """Fired when the agent outputs Final Answer."""
        log_text = getattr(finish, "log", "") or ""
        thought, _ = _split_thought(log_text)
        final = finish.return_values.get("output", "")

        if thought:
            self._log.add_step(
                module="Supervisor",
                submodule="Thought",
                prompt="",
                response=thought,
            )
        if final:
            self._log.add_step(
                module="Supervisor",
                submodule="FinalAnswer",
                prompt="",
                response=final,
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
            vehicle_retriever: Optional["VehicleModelRetriever"] = None,
            target_listings: int = NUM_TARGET_LISTINGS,
            max_iterations: int = MAX_DECISION_ITERATIONS
    ):
        self.llm_gateway = llm_gateway
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
        self._no_vehicles_found: bool = False   # set True when RAG returns nothing
        self._inexact_model_note: Optional[str] = None  # set when user asked for unknown model

        # Use the pre-built retriever if provided (avoids reconnect on every request)
        self._vehicle_retriever: Optional[VehicleModelRetriever] = vehicle_retriever

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
            handle_parsing_errors=(
                "Your output contained both an Action and a Final Answer, or an Observation line. "
                "Output ONLY one of: (a) Thought + Action + Action Input, OR (b) Thought + Final Answer. "
                "Never write Observation lines yourself — they are provided automatically. Try again."
            ),
            callbacks=[SupervisorLogCallback(self.action_log)],
        )

    # ========================================================================
    # Action Methods  (called by LangChain tool wrappers in tools.py)
    # ========================================================================
    def _get_vehicle_retriever(self) -> VehicleModelRetriever:
        """Lazily initialize the VehicleModelRetriever with Pinecone + LLM."""
        if self._vehicle_retriever is None:
            print("\n🔌 Connecting to Pinecone...")
            pinecone_index = get_pinecone_index()
            embedding_gateway = EmbeddingGateway.get_instance(api_key=OPENAI_API_KEY)
            self._vehicle_retriever = VehicleModelRetriever(
                pinecone_index=pinecone_index,
                embedding_gateway=embedding_gateway,
                llm_gateway=self.llm_gateway,
            )
        return self._vehicle_retriever

    # -----------------------------------------------------------------------
    # Generic non-model words that may follow a make name in a query.
    # If the word after the make is in this set it does NOT count as a
    # model-name request, so we skip the inexact-match check.
    _NON_MODEL_WORDS: frozenset = frozenset({
        # price / budget words
        "under", "below", "max", "maximum", "budget", "price",
        # year / time words
        "from", "since", "after", "newer", "onwards", "onward",
        # connectors / articles
        "and", "or", "the", "a", "an", "with", "for", "of",
        # body-type / category words (not model names)
        "suv", "sedan", "saloon", "truck", "pickup", "van", "minivan", "mpv",
        "hatchback", "coupe", "convertible", "wagon", "estate", "crossover",
        "hybrid", "electric", "diesel", "gas", "phev",
        # descriptors
        "used", "new", "reliable", "family", "luxury", "economy",
        "compact", "midsize", "full", "large", "small", "cheap", "affordable",
    })

    def _detect_inexact_model(self, query: str, vehicles: list) -> Optional[str]:
        """
        Return the unmatched token (as typed by the user) when the query names
        a make+model combination not present in the returned vehicles, or when
        the brand name itself appears to be a typo of a returned make.
        Returns None when the query matches the results well enough.

        Algorithm (pure Python, no LLM):
          1. Collect the (make, model) pairs of all returned vehicles.
          2. Exact-make pass: for each returned make that appears verbatim in
             the query, extract the word immediately following it.  If that word
             looks like a model name (not a generic term) and is not a substring
             of any returned model for that make → return "<make> <queried_word>".
          3. Fuzzy-brand pass: for each query word that did NOT match any make
             exactly, check if it is close (≥ 0.75 similarity) to a returned make.
             If so, the user typed a brand typo (e.g. "hunda" → "honda") — return
             the original query token(s) so the caller can surface a note.
        """
        import re as _re
        from difflib import get_close_matches

        if not vehicles or not query:
            return None

        query_lower = query.lower()
        returned_pairs = [
            (v.get("make", "").lower().strip(), v.get("model", "").lower().strip())
            for v in vehicles
        ]
        returned_makes = {make for make, _ in returned_pairs if make}

        # --- Pass 1: exact make match, check model word ---
        for make in returned_makes:
            m = _re.search(
                r"\b" + _re.escape(make) + r"\s+([a-zA-Z0-9]+)",
                query_lower,
            )
            if not m:
                continue
            queried_word = m.group(1).lower()
            if queried_word in self._NON_MODEL_WORDS:
                continue
            model_matched = any(
                queried_word in model
                for ret_make, model in returned_pairs
                if ret_make == make
            )
            if not model_matched:
                return f"{make} {queried_word}"

        # --- Pass 2: fuzzy brand match — catches brand typos like "hunda civic" ---
        query_words = _re.findall(r"[a-zA-Z0-9]+", query_lower)
        for i, word in enumerate(query_words):
            if word in returned_makes:
                continue  # exact match already handled above
            close = get_close_matches(word, returned_makes, n=1, cutoff=0.75)
            if not close:
                continue
            # word is a typo of a make; include the following model word if present
            if i + 1 < len(query_words) and query_words[i + 1] not in self._NON_MODEL_WORDS:
                return f"{word} {query_words[i + 1]}"
            return word

        return None

    def _action_search_vehicle_models(self) -> str:
        """Search for matching vehicle models using the RAG pipeline (Pinecone + LLM)."""
        query = (
            self.user_requirements.get("full_query", "")
            if self.user_requirements else ""
        )
        if not query:
            return "Error: No user query available to search vehicle models"

        print(f"\n🔍 Searching vehicle models via RAG for: '{query}'")
        try:
            retriever = self._get_vehicle_retriever()
            result, _rag_details = retriever.search_vehicle_models(query=query)
            self.vehicle_models = result.get("vehicles", [])

            # Pure-Python inexact-model detection (does not rely on LLM flag)
            unmatched_token = self._detect_inexact_model(query, self.vehicle_models) if self.vehicle_models else None
            if unmatched_token:
                self._inexact_model_note = unmatched_token

            # Log full vehicle model details in the trace
            vehicle_detail_lines = []
            for v in self.vehicle_models:
                vehicle_detail_lines.append(
                    f"{v.get('make', '')} {v.get('model', '')} "
                    f"({v.get('body_type', '')}, {v.get('years', '')}) "
                    f"| match: {float(v.get('match_score', 0)):.2f} "
                    f"| {v.get('match_reason', '')}"
                )
            self.action_log.add_step(
                module="SearchPipeline",
                submodule="VehicleModelRetriever",
                prompt=query,
                response="\n".join(vehicle_detail_lines) if vehicle_detail_lines else "No models found",
            )
        except Exception as e:
            return f"Error during vehicle model search: {e}"

        if not self.vehicle_models:
            self._no_vehicles_found = True
            return (
                f"No vehicle models found for query: '{query}'. "
                "The RAG index may not contain matching models. "
                "Consider broadening the search terms."
            )

        self.actions_taken.add("search_vehicle_models")
        return f"Found {len(self.vehicle_models)} vehicle models: " + ", ".join(
            f"{v.get('make', '')} {v.get('model', '')}" for v in self.vehicle_models
        )

    def _action_retrieve_listings(self) -> str:
        """Retrieve listings from CSV using the found vehicle models."""
        if not self.vehicle_models:
            return "Error: Cannot retrieve listings without vehicle models"

        print("Retrieving listings from database...")

        # Build a VehicleModelsResult from the raw vehicle dicts
        raw_result = {"vehicles": self.vehicle_models, "explanation": ""}
        vehicles_result = VehicleModelsResult.from_raw_result(
            query=self.user_requirements.get("full_query", "") if self.user_requirements else "",
            raw_result=raw_result,
        )


        retriever = ListingsRetriever()

        # Parse user constraints for the fallback tiers in retrieve_listings
        reqs = self.user_requirements or {}
        try:
            year_min = int(reqs.get("year_min", 0)) or None
        except (TypeError, ValueError):
            year_min = None
        try:
            price_max = float(reqs.get("max_price", 0)) or None
        except (TypeError, ValueError):
            price_max = None

        listing_objects = retriever.retrieve_listings(
            vehicle_models_result=vehicles_result,
            year_min=year_min,
            price_max=price_max,
        )

        # Convert VehicleListing objects back to the dict format the rest of supervisor expects
        results_by_vehicle = {}
        for listing in listing_objects:
            d = listing.to_dict() if hasattr(listing, "to_dict") else vars(listing)
            key = f"{d.get('manufacturer', '')} {d.get('model', '')}" .strip()
            if key not in results_by_vehicle:
                results_by_vehicle[key] = []
            results_by_vehicle[key].append(d)

        self.listings = {"results": []}
        for vehicle_model, listings_list in results_by_vehicle.items():
            # Find matching vehicle meta from self.vehicle_models
            v_meta = {}
            for v in self.vehicle_models:
                if vehicle_model.lower() == f"{v.get('make','')} {v.get('model','')}" .lower().strip():
                    v_meta = v
                    break
            self.listings["results"].append({"vehicle": v_meta, "listings": listings_list})

        total = sum(len(r.get("listings", [])) for r in self.listings["results"])

        # Log ListingsRetriever step with per-model counts
        per_model = ", ".join(
            f"{r['vehicle'].get('make', '')} {r['vehicle'].get('model', '')}: {len(r['listings'])} listings"
            for r in self.listings["results"]
        )
        self.action_log.add_step(
            module="SearchPipeline",
            submodule="ListingsRetriever",
            prompt=", ".join(
                f"{v.get('make', '')} {v.get('model', '')}" for v in self.vehicle_models
            ),
            response=f"Total: {total} listings | {per_model}",
        )

        # Run DecisionAgent to score and rank listings and log the results
        try:
            from agents.search_agent.decision_agent import DecisionAgent
            decision_agent = DecisionAgent()
            scored = decision_agent.get_scored_listings(
                listings=listing_objects,
                vehicle_models_result=vehicles_result,
            )
            if scored:
                ranking_lines = []
                for i, sl in enumerate(scored[:self.target_listings], 1):
                    reasons = ", ".join(sl.reasons) if sl.reasons else "no specific reasons"
                    ranking_lines.append(
                        f"#{i}: {sl.listing.manufacturer} {sl.listing.model} "
                        f"({sl.listing.year}, ${sl.listing.price}) "
                        f"| score: {sl.final_score:.2f} "
                        f"| {reasons}"
                    )
                self.action_log.add_step(
                    module="SearchPipeline",
                    submodule="DecisionAgent",
                    prompt=f"Score and rank {len(listing_objects)} listings",
                    response="\n".join(ranking_lines),
                )
        except Exception:
            pass  # DecisionAgent is optional — don't break the pipeline

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

        # ── Final DecisionAgent re-ranking with enriched data ────────────────
        try:
            from agents.search_agent.decision_agent import DecisionAgent
            from agents.utils.contracts import VehicleListing, VehicleModel, VehicleModelsResult

            # Collect all enriched listings as VehicleListing objects
            enriched_listings: list = []
            for group in self.processed_results.get("results", []):
                for listing_dict in group.get("listings", []):
                    enriched_listings.append(VehicleListing.from_dict(listing_dict))

            if enriched_listings and self.vehicle_models:
                raw_result = {"vehicles": self.vehicle_models, "explanation": ""}
                vehicles_result = VehicleModelsResult.from_raw_result(
                    query=self.user_requirements.get("full_query", "") if self.user_requirements else "",
                    raw_result=raw_result,
                )

                decision_agent = DecisionAgent()
                scored = decision_agent.get_scored_listings(
                    listings=enriched_listings,
                    vehicle_models_result=vehicles_result,
                )

                if scored:
                    # Log final ranking to execution trace
                    ranking_lines = []
                    for i, sl in enumerate(scored, 1):
                        reasons = ", ".join(sl.reasons) if sl.reasons else "no specific reasons"
                        ranking_lines.append(
                            f"#{i}: {sl.listing.manufacturer} {sl.listing.model} "
                            f"({sl.listing.year}, ${sl.listing.price}) "
                            f"| score: {sl.final_score:.2f} "
                            f"| {reasons}"
                        )
                    self.action_log.add_step(
                        module="SearchPipeline",
                        submodule="DecisionAgent (final re-ranking)",
                        prompt=f"Re-rank {len(enriched_listings)} fully enriched listings",
                        response="\n".join(ranking_lines),
                    )

                    # Re-sort the results groups so best listings appear first
                    score_map: dict = {}
                    for sl in scored:
                        key = str(getattr(sl.listing, "id", ""))
                        score_map[key] = sl.final_score

                    for group in self.processed_results.get("results", []):
                        group["listings"].sort(
                            key=lambda d: score_map.get(str(d.get("id", "")), 0.0),
                            reverse=True,
                        )

                    print(f"\n🏆 Final ranking complete — top listing: "
                          f"{scored[0].listing.manufacturer} {scored[0].listing.model} "
                          f"(score: {scored[0].final_score:.2f})")

        except Exception as e:
            print(f"⚠️  Final DecisionAgent ranking skipped: {e}")

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

        return self._run_agent_loop()

    def run_headless(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the supervisor without any stdin prompts.
        Used by the FastAPI /api/execute endpoint.

        Args:
            requirements: A dict with at least:
                - ``max_price``  (str or number)
                - ``year_min``   (str or number)
                - ``description`` (str, optional)
                - ``full_query``  (str, optional)

        Returns:
            Same dict as run(): {"results", "stats", "steps"}
        """
        print("\n" + "=" * 60)
        print("🎯 AUTONOMOUS SUPERVISOR: Starting mission (headless / API)")
        print(f"Goal: Find {self.target_listings} complete listings")
        print("=" * 60)

        self.user_requirements = requirements
        self.actions_taken.add("collect_requirements")

        # Build full_query if not already present
        if not self.user_requirements.get("full_query"):
            parts = []
            if requirements.get("max_price"):
                parts.append(f"max price: {requirements['max_price']}")
            if requirements.get("year_min"):
                parts.append(f"min year: {requirements['year_min']}")
            if requirements.get("description"):
                parts.append(requirements["description"])
            self.user_requirements["full_query"] = " ".join(parts)

        return self._run_agent_loop()

    def _run_agent_loop(self) -> Dict[str, Any]:
        """Shared ReAct loop used by both run() and run_headless()."""
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

        # ── Early exit: RAG found no matching vehicle models ─────────────────
        if self._no_vehicles_found:
            return {
                "results": [],
                "stats": {},
                "steps": self.action_log.get_steps(),
                "error": "no_vehicles_found",
            }

        self._print_summary()

        result = {
            "results": self.processed_results.get("results", []) if self.processed_results else [],
            "stats": self.processed_results.get("stats", {}) if self.processed_results else {},
            "steps": self.action_log.get_steps(),
        }
        if self._inexact_model_note:
            result["inexact_model_note"] = self._inexact_model_note
        return result

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
