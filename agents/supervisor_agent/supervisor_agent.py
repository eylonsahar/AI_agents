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

from typing import Dict, Any, Optional, List

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agents.supervisor_agent.user_communication import UserCommunication
from agents.supervisor_agent.tools import make_supervisor_tools
from agents.search_agent.search_pipeline import SearchPipeline, create_pipeline
from agents.search_agent.rag_retrieval import get_pinecone_index
from agents.field_agent.field_agent import FieldAgent
from gateways.llm_gateway import LLMGateway
from gateways.embedding_gateway import EmbeddingGateway
from agents.action_log import ActionLog
from agents.prompts import SUPERVISOR_REACT_PROMPT

from dotenv import load_dotenv
import os
import json
from config import MAX_DECISION_ITERATIONS, NUM_TARGET_LISTINGS, MAX_RECOMMENDED_MODELS

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
            embedding_gateway: EmbeddingGateway = None,
            pinecone_index = None,
            target_listings: int = NUM_TARGET_LISTINGS,
            max_iterations: int = MAX_DECISION_ITERATIONS
    ):
        self.llm_gateway = llm_gateway
        self.embedding_gateway = embedding_gateway or EmbeddingGateway.get_instance(api_key=OPENAI_API_KEY)
        self.pinecone_index = pinecone_index
        self.user_comm = UserCommunication()

        self.target_listings = target_listings
        self.max_iterations = max_iterations

        # Shared action log — passed down to SearchPipeline and FieldAgent
        self.action_log = ActionLog()

        # State tracking
        self.user_requirements = None
        self.vehicle_models = None
        self.pipeline_result = None  # Result from SearchPipeline
        self.listings = None  # Formatted listings for FieldAgent
        self.processed_results = None
        self._no_results_found = False  # Flag for empty search results
        self._mission_complete = False  # Flag to stop ReAct loop

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
    def _action_search_vehicles(self, query: str) -> str:
        """Run the full search pipeline: find vehicle models and retrieve listings."""
        if not self.pinecone_index:
            return "Error: Pinecone index not initialized"

        print(f"\n🔍 Running search pipeline for: '{query}'...")

        # Create and run the search pipeline
        pipeline = SearchPipeline(
            pinecone_index=self.pinecone_index,
            embedding_gateway=self.embedding_gateway,
            llm_gateway=self.llm_gateway,
            action_log=self.action_log  # Share the action log
        )
        
        self.pipeline_result = pipeline.search(query)
        
        # Store vehicle models
        if self.pipeline_result.vehicle_models_result:
            self.vehicle_models = [v.to_dict() for v in self.pipeline_result.vehicle_models_result.vehicles]
        
        # Log vehicle models with reasoning
        self._print_vehicle_models()
        
        # Format listings for FieldAgent (group by vehicle model)
        self.listings = self._format_listings_for_field_agent()
        
        num_models = len(self.vehicle_models) if self.vehicle_models else 0
        num_listings = len(self.pipeline_result.scored_listings) if self.pipeline_result.scored_listings else 0
        
        self.actions_taken.add("search_vehicles")
        
        # Handle no results found
        if num_models == 0 or num_listings == 0:
            self._no_results_found = True
            return (
                f"NO RESULTS FOUND for query: '{query}'. "
                f"You must call complete_mission to inform the user that no vehicles were found "
                f"and suggest they try a different search with broader criteria."
            )
        
        return f"Found {num_models} vehicle models and {num_listings} listings"
    
    def _print_vehicle_models(self) -> None:
        """Print the recommended vehicle models with their reasoning."""
        if not self.pipeline_result or not self.pipeline_result.vehicle_models_result:
            return
        
        print("\n" + "=" * 60)
        print("🚗 RECOMMENDED VEHICLE MODELS")
        print("=" * 60)
        
        for i, vehicle in enumerate(self.pipeline_result.vehicle_models_result.vehicles, 1):
            print(f"\n[{i}] {vehicle.make} {vehicle.model}")
            print(f"    Body Type: {vehicle.body_type}")
            print(f"    Years: {vehicle.years}")
            print(f"    Match Score: {vehicle.match_score}")
            print(f"    Reason: {vehicle.match_reason[:200]}..." if len(vehicle.match_reason) > 200 else f"    Reason: {vehicle.match_reason}")
        
        print("\n" + "=" * 60)

    def _format_listings_for_field_agent(self) -> Dict[str, Any]:
        """Format pipeline results into the structure expected by FieldAgent."""
        if not self.pipeline_result or not self.pipeline_result.scored_listings:
            return {"results": []}
        
        # Group listings by vehicle model
        groups: Dict[str, Dict] = {}
        
        for scored in self.pipeline_result.scored_listings:
            listing = scored.listing
            vehicle = scored.vehicle_model
            key = f"{vehicle.make}_{vehicle.model}"
            
            if key not in groups:
                groups[key] = {
                    "vehicle": {
                        "make": vehicle.make,
                        "model": vehicle.model,
                        "match_reason": vehicle.match_reason
                    },
                    "listings": []
                }
            
            # Convert listing to dict format
            listing_dict = listing.to_dict()
            groups[key]["listings"].append(listing_dict)
        
        return {"results": list(groups.values())}

    def _action_process_listings(self) -> str:
        """Delegate to FieldAgent for listing completion and scheduling."""
        if not self.listings or not self.listings.get("results"):
            return "Error: Cannot process without listings. Run search_vehicles first."

        print("\n🤖 Delegating to autonomous field agent...")

        field_agent = FieldAgent(
            ads_list=self.listings,
            action_log=self.action_log,  # shared log
        )
        self.processed_results = field_agent.process_listings()

        # Final ranking with DecisionAgent after FieldAgent completes
        self._final_ranking()

        stats = self.processed_results.get("stats", {})
        self.actions_taken.add("process_listings")
        return f"Field agent completed: {stats.get('completed_listings', 0)} listings ready with meetings scheduled"
    
    def _final_ranking(self) -> None:
        """Re-rank listings with DecisionAgent after FieldAgent fills in data."""
        if not self.processed_results or not self.pipeline_result:
            return
        
        from agents.search_agent.decision_agent import DecisionAgent
        from agents.utils.contracts import VehicleListing
        
        print("\n📊 Final ranking with DecisionAgent...")
        
        # Convert processed listings back to VehicleListing objects
        all_listings: List[VehicleListing] = []
        for group in self.processed_results.get("results", []):
            for listing_dict in group.get("listings", []):
                listing = VehicleListing.from_dict(listing_dict)
                all_listings.append(listing)
        
        if not all_listings:
            return
        
        # Re-score with DecisionAgent
        decision_agent = DecisionAgent()
        scored_listings = decision_agent.get_scored_listings(
            listings=all_listings,
            vehicle_models_result=self.pipeline_result.vehicle_models_result
        )
        
        # Print final ranking
        print("\n" + "=" * 60)
        print("🏆 FINAL RANKED LISTINGS")
        print("=" * 60)
        
        for i, scored in enumerate(scored_listings[:10], 1):  # Top 10
            listing = scored.listing
            print(f"\n[{i}] {scored.vehicle_model.make} {scored.vehicle_model.model}")
            print(f"    Price: ${listing.price:,}" if listing.price else "    Price: N/A")
            print(f"    Year: {int(listing.year)}" if listing.year else "    Year: N/A")
            print(f"    Final Score: {scored.final_score:.2f}")
            print(f"    Reasons: {', '.join(scored.reasons[:2])}")
        
        print("\n" + "=" * 60)
        
        # Update processed_results with ranked order
        self._reorder_results_by_score(scored_listings)
    
    def _reorder_results_by_score(self, scored_listings: List) -> None:
        """Reorder the processed results by final score."""
        # Create a mapping of listing_id to score
        score_map = {scored.listing.id: scored.final_score for scored in scored_listings}
        
        # Sort listings within each group by score
        for group in self.processed_results.get("results", []):
            group["listings"].sort(
                key=lambda l: score_map.get(l.get("id"), 0),
                reverse=True
            )

    def _action_complete_mission(self) -> str:
        """Present final results to the user."""
        self._mission_complete = True
        
        # Handle no results case
        if self._no_results_found:
            self._present_no_results()
            self.actions_taken.add("complete_mission")
            return "Informed user that no vehicles were found. Suggested trying a different search. Mission complete! You MUST now output 'Final Answer: Mission complete' to end."
        
        if not self.processed_results:
            return "Error: No results to present. Run process_listings first."

        print("\n📋 Presenting results to user...")

        meetings_list = self._extract_meetings(self.processed_results)
        self.user_comm.return_vehicle_ads(
            listings_groups=self.processed_results["results"],
            meetings_list=meetings_list,
            action_log=self.action_log.get_steps(),
        )

        self.actions_taken.add("complete_mission")
        return "Results presented to user. Mission complete! You MUST now output 'Final Answer: Mission complete' to end."
    
    def _present_no_results(self) -> None:
        """Present a message to the user when no vehicles were found."""
        print("\n" + "=" * 60)
        print("😔 NO VEHICLES FOUND")
        print("=" * 60)
        print("\nI couldn't find any vehicles matching your criteria.")
        print("\n💡 SUGGESTIONS:")
        print("   • Try increasing your maximum budget")
        print("   • Consider older model years")
        print("   • Broaden your vehicle type preferences")
        print("   • Try different keywords in your description")
        print("\nPlease run the search again with different criteria.")
        print("=" * 60 + "\n")

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

        # Initialize Pinecone if not provided
        if not self.pinecone_index:
            print("\n[Initialization] Connecting to Pinecone...")
            self.pinecone_index = get_pinecone_index()

        # Collect requirements from user (no LLM call)
        print("\n[Initialization] Collecting user requirements...")
        self.user_requirements = self.user_comm.get_vehicle_request()
        self.actions_taken.add("collect_requirements")

        # Build the search query from user requirements
        search_query = self.user_requirements.get('full_query', self.user_requirements.get('description', ''))

        # Build the input prompt for the ReAct agent
        agent_input = (
            f"Find vehicle listings and schedule meetings for the user.\n\n"
            f"User requirements:\n"
            f"  Max price: ${self.user_requirements.get('max_price', 'N/A')}\n"
            f"  Min year: {self.user_requirements.get('year_min', 'N/A')}\n"
            f"  Description: {self.user_requirements.get('description', 'N/A')}\n"
            f"  Full query: {search_query}\n\n"
            f"Execute the tools in order:\n"
            f"1. search_vehicles - with the user's query to find matching vehicles and listings\n"
            f"2. process_listings - to fill missing data and schedule meetings\n"
            f"3. complete_mission - to present results to the user"
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
