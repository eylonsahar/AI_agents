"""
FastAPI server — AItzik AI Car Agent API
==========================================

Exposes four required HTTP endpoints:
    GET  /api/team_info
    GET  /api/agent_info
    GET  /api/model_architecture
    POST /api/execute
"""

import os
import sys
import json
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# Suppress LangChain's verbose OUTPUT_PARSING_FAILURE console message — the
# error is already handled gracefully by handle_parsing_errors on each executor.
logging.getLogger("langchain_core.output_parsers").setLevel(logging.ERROR)
logging.getLogger("langchain.agents").setLevel(logging.ERROR)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# ── project root on path ───────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

# ── agent imports ──────────────────────────────────────────────────────────
from agents.supervisor_agent.supervisor_agent import AgentSupervisor
from agents.search_agent.vehicle_model_retriever import VehicleModelRetriever
from agents.search_agent.rag_retrieval import get_pinecone_index
from gateways.llm_gateway import LLMGateway
from gateways.embedding_gateway import EmbeddingGateway

# ── Singleton RAG components (initialized once on startup) ─────────────────
_vehicle_retriever: Optional["VehicleModelRetriever"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources once when the server starts."""
    global _vehicle_retriever
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            print("\n🔌 [Startup] Connecting to Pinecone...")
            pinecone_index = get_pinecone_index()
            embedding_gateway = EmbeddingGateway.get_instance(api_key=api_key)
            llm_gateway = LLMGateway.get_instance(api_key=api_key)
            _vehicle_retriever = VehicleModelRetriever(
                pinecone_index=pinecone_index,
                embedding_gateway=embedding_gateway,
                llm_gateway=llm_gateway,
            )
            print("✅ [Startup] Pinecone ready.")
        except Exception as e:
            print(f"⚠️  [Startup] Could not init Pinecone: {e}")
    else:
        print("⚠️  [Startup] OPENAI_API_KEY not set — Pinecone not initialized.")
    yield  # server is running
    # (cleanup here if needed)


# ===========================================================================
# App
# ===========================================================================

app = FastAPI(
    title="AItzik — AI Car Agent",
    description="Multi-agent system for vehicle search and meeting scheduling",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend at the root
FRONTEND_DIR = os.path.join(ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "AItzik API is running. Use /api/* endpoints."})


# ===========================================================================
# Pydantic schemas
# ===========================================================================

class ExecuteRequest(BaseModel):
    prompt: str
    max_price: Optional[str] = None
    year_min: Optional[str] = None
    description: Optional[str] = None


class StepSchema(BaseModel):
    module: str
    prompt: Any
    response: Any


class ExecuteResponse(BaseModel):
    status: str                     # "ok" | "error"
    error: Optional[str]
    response: Optional[str]
    steps: List[Dict[str, Any]]


# ===========================================================================
# Helpers
# ===========================================================================

ARCHITECTURE_PNG = os.path.join(ROOT, "model_architecture.png")


def _check_is_car_search(prompt: str, description: str, llm_gateway) -> bool:
    """
    Ask the LLM whether the user's request is a car/vehicle search.
    Returns True if it is, False otherwise.
    """
    text = f"{prompt} {description}".strip()
    try:
        response = llm_gateway.call_llm(
            system_prompt=(
                "You are a request classifier. "
                "Answer ONLY with the single word 'yes' or 'no'. "
                "'yes' = the request is about finding / buying a used car or vehicle. "
                "'no' = the request is about anything else."
            ),
            user_message=f"Is this a car search request? Request: '{text}'",
        )
        return "yes" in response.strip().lower()[:10]
    except Exception:
        return True   # fail open: let the agent try


def _format_results_as_text(results: List[Dict]) -> str:
    """Convert processed agent results into a readable final response string."""
    if not results:
        return (
            "I couldn't find any vehicles matching your criteria. "
            "Try adjusting your budget, year range, or description."
        )

    lines = ["🌟 AItzik's TOP RECOMMENDATIONS\n"]
    ad_counter = 1

    for group in results:
        vehicle = group.get("vehicle", {})
        listings = group.get("listings", [])
        make = vehicle.get("make", "").upper()
        model = vehicle.get("model", "").upper()
        reason = vehicle.get("match_reason", "Matches your criteria.")

        lines.append(f"━━ {make} {model} ━━")
        lines.append(f"💡 Why this model: {reason}\n")

        for listing in listings:
            listing_data = listing if "id" in listing else listing.get("listing_data", listing)
            lines.append(f"  Option #{ad_counter}:")

            skip_keys = {"id", "posting_date", "manufacturer", "model", "_posting_dt"}
            for key, value in listing_data.items():
                if key not in skip_keys and value not in (None, "", "Not Provided"):
                    display_key = key.replace("_", " ").title()
                    lines.append(f"    • {display_key}: {value}")

            meetings = listing_data.get("meetings", [])
            if meetings:
                lines.append("    📅 Schedule a viewing:")
                for slot in meetings:
                    lines.append(f"       🔗 {slot.get('slot', '')}: {slot.get('url', '')}")

            lines.append("")
            ad_counter += 1

    lines.append("Click the links above to add viewings to your calendar!")
    return "\n".join(lines)


def _normalize_steps(raw_steps: List[Dict]) -> List[Dict]:
    """
    Ensure every step conforms to the required schema:
        {module, prompt, response}
    The internal ActionLog also stores 'submodule'; we keep it as extra info
    inside the module field (e.g. "FieldAgent / MockSeller/GetData").
    """
    normalized = []
    for step in raw_steps:
        module = step.get("module", "Unknown")
        submodule = step.get("submodule", "")
        if submodule:
            module = f"{module} / {submodule}"

        normalized.append(
            {
                "module": module,
                "prompt": step.get("prompt", ""),
                "response": step.get("response", ""),
            }
        )
    return normalized


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/api/team_info")
def team_info():
    """Return student / team details."""
    return {
        "group_batch_order_number": "1_3",
        "team_name": "AItzik",
        "students": [
            {"name": "Eylon Sahar", "email": "eylon.sahar@campus.technion.ac.il"},
            {"name": "Aviv Rabi",   "email": "aviv.rabi@campus.technion.ac.il"},
            {"name": "Ron Bartal",  "email": "ron.bartal@campus.technion.ac.il"},
        ],
    }


@app.get("/api/agent_info")
def agent_info():
    """Return agent meta-information, purpose, prompt templates, and examples."""
    return {
        "description": (
            "AItzik is a multi-agent system that autonomously finds used vehicles matching "
            "your requirements, fills in missing listing data by contacting mock sellers, "
            "and schedules Google Calendar meeting links so you can view each car."
        ),
        "purpose": (
            "Help users find the best used vehicles on the market by automating: "
            "(1) semantic vehicle-model search via RAG + Pinecone, "
            "(2) live listings retrieval from a structured CSV database, "
            "(3) data enrichment via a Field Agent that interviews mock sellers, and "
            "(4) meeting scheduling with Google Calendar deep links."
        ),
        "prompt_template": {
            "template": (
                "I'm looking for a {vehicle_description}. "
                "My maximum budget is ${max_price} and I want a car from {year_min} or newer. "
                "{optional_preferences}"
            ),
            "fields": {
                "vehicle_description": "Natural language description, e.g. 'reliable family SUV'",
                "max_price": "Maximum price in USD, e.g. '20000'",
                "year_min": "Minimum model year, e.g. '2018'",
                "optional_preferences": "Any additional preferences, e.g. 'preferably silver, low mileage'",
            },
        },
        "prompt_examples": [
            {
                "prompt": "I need a reliable family SUV, max price $20000, minimum year 2018",
                "full_response": (
                    "🌟 AItzik's TOP RECOMMENDATIONS\n\n"
                    "━━ KIA SPORTAGE ━━\n"
                    "💡 Why this model: Excellent reliability and space for families, within budget.\n\n"
                    "  Option #1:\n"
                    "    • Price: 18500\n"
                    "    • Year: 2019\n"
                    "    • Condition: Good\n"
                    "    • Mileage: 62000\n"
                    "    • Accident: none\n"
                    "    • Paint Color: silver\n"
                    "    • State: CA\n"
                    "    📅 Schedule a viewing:\n"
                    "       🔗 2026-03-05 14:00: https://www.google.com/calendar/render?..."
                ),
                "steps": [
                    {
                        "module": "Supervisor / DecisionMaking",
                        "prompt": "Find 4 complete vehicle listings and schedule meetings...",
                        "response": "Thought: I should search for vehicle models first.\nAction: search_vehicle_models\nAction Input: \"\"\nObservation: Loaded 3 vehicle models from JSON"
                    },
                    {
                        "module": "Supervisor / DecisionMaking",
                        "prompt": "...(continued ReAct loop)...",
                        "response": "Thought: Now I should retrieve listings.\nAction: retrieve_listings\nAction Input: \"\"\nObservation: Retrieved 9 listings"
                    },
                    {
                        "module": "Supervisor / DecisionMaking",
                        "prompt": "...(continued)...",
                        "response": "Action: process_listings"
                    },
                    {
                        "module": "FieldAgent / DecisionMaking",
                        "prompt": "Process all vehicle listings...",
                        "response": "Thought: Listing 7391 needs mileage and accident data.\nAction: fill_missing_data\nAction Input: {\"listing_id\": \"7391\", \"fields_to_request\": [\"mileage\", \"accident\"]}"
                    },
                    {
                        "module": "FieldAgent / MockSeller/GetData",
                        "prompt": "You are a private individual selling a used car...",
                        "response": "mileage = 62000\naccident = none\npaint_color = silver\nstate = ca\ncondition = good"
                    },
                    {
                        "module": "FieldAgent / MockSeller/Scheduling",
                        "prompt": "Generate exactly 2 available slots...",
                        "response": "2026-03-05 14:00\n2026-03-08 11:00"
                    }
                ]
            },
            {
                "prompt": "Looking for a sporty coupe under $15000, year 2016 or newer",
                "full_response": (
                    "I couldn't find any vehicles matching your criteria exactly. "
                    "The RAG search did not return sporty coupes in the current database. "
                    "Try widening your budget or choosing a more common body type such as sedan or SUV."
                ),
                "steps": [
                    {
                        "module": "Supervisor / DecisionMaking",
                        "prompt": "Find 4 complete vehicle listings...",
                        "response": "Action: search_vehicle_models"
                    },
                    {
                        "module": "Supervisor / DecisionMaking",
                        "prompt": "...",
                        "response": "Warning: JSON loaded but 'vehicles' list is empty"
                    }
                ]
            }
        ]
    }


@app.get("/api/model_architecture")
def model_architecture():
    """Serve the model architecture diagram as a PNG image."""
    if not os.path.isfile(ARCHITECTURE_PNG):
        raise HTTPException(
            status_code=404,
            detail=f"model_architecture.png not found at {ARCHITECTURE_PNG}"
        )
    return FileResponse(
        path=ARCHITECTURE_PNG,
        media_type="image/png",
        filename="model_architecture.png",
    )


@app.post("/api/execute", response_model=ExecuteResponse)
def execute(body: ExecuteRequest):
    """
    Run the full multi-agent pipeline with the given prompt.

    The body may include:
    - ``prompt``      (required) — free-text request
    - ``max_price``   (optional) — if not provided, extracted from prompt
    - ``year_min``    (optional) — if not provided, extracted from prompt
    - ``description`` (optional) — additional preferences
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return ExecuteResponse(
                status="error",
                error="OPENAI_API_KEY not configured on the server.",
                response=None,
                steps=[],
            )

        # ── Parse requirements from body or fallback to prompt ────────────
        prompt_text = body.prompt.strip()
        if not prompt_text:
            return ExecuteResponse(
                status="error",
                error="Prompt cannot be empty.",
                response=None,
                steps=[],
            )

        max_price = body.max_price or _extract_price(prompt_text)
        year_min = body.year_min or _extract_year(prompt_text)
        description = body.description or prompt_text

        if not max_price:
            return ExecuteResponse(
                status="error",
                error=(
                    "Could not determine your maximum price. "
                    "Please include a budget, e.g. 'max price $20000', "
                    "or use the max_price field."
                ),
                response=None,
                steps=[],
            )

        if not year_min:
            return ExecuteResponse(
                status="error",
                error=(
                    "Could not determine a minimum year for the vehicle. "
                    "Please include a year, e.g. 'from 2018 onwards', "
                    "or use the year_min field."
                ),
                response=None,
                steps=[],
            )

        requirements = {
            "max_price": max_price,
            "year_min": year_min,
            "description": description,
        }

        # ── Intent check: reject non-car requests immediately ──────────────
        llm_gateway = LLMGateway.get_instance(api_key=api_key)
        if not _check_is_car_search(prompt_text, description, llm_gateway):
            return ExecuteResponse(
                status="error",
                error=(
                    "Your request doesn't seem to be about finding a car. "
                    "Please describe the vehicle you're looking for, e.g. "
                    "'reliable family SUV under $20,000 from 2018 or newer'."
                ),
                response=None,
                steps=[],
            )

        # ── Run the agent ─────────────────────────────────────────────────
        supervisor = AgentSupervisor(
            llm_gateway=llm_gateway,
            vehicle_retriever=_vehicle_retriever,  # pre-built at startup
        )
        result = supervisor.run_headless(requirements=requirements)

        # ── Handle no-vehicles-found gracefully ────────────────────────────
        if result.get("error") == "no_vehicles_found":
            query = requirements.get("description", prompt_text)
            return ExecuteResponse(
                status="error",
                error=(
                    f"I couldn't find any vehicles matching your search: '{query}'. "
                    "This may be because the description doesn't match cars in our database. "
                    "Try different keywords — for example: 'SUV', 'sedan', 'pickup truck', "
                    "or specific makes like 'Toyota', 'Honda', 'Ford'."
                ),
                response=None,
                steps=_normalize_steps(result.get("steps", [])),
            )

        raw_steps = result.get("steps", [])
        results = result.get("results", [])

        response_text = _format_results_as_text(results)
        normalized_steps = _normalize_steps(raw_steps)

        return ExecuteResponse(
            status="ok",
            error=None,
            response=response_text,
            steps=normalized_steps,
        )

    except Exception as exc:
        err_detail = str(exc)
        traceback.print_exc()
        return ExecuteResponse(
            status="error",
            error=err_detail,
            response=None,
            steps=[],
        )


# ===========================================================================
# Extraction helpers
# ===========================================================================

import re as _re


def _extract_price(text: str) -> Optional[str]:
    """Try to extract a dollar amount from a natural-language prompt."""
    # Patterns like: $20000, 20,000, 20k, max price 20000, budget 15000
    patterns = [
        r"\$\s*([\d,]+(?:\.\d+)?)\s*k?\b",
        r"(?:max(?:imum)?\s+(?:price|budget)|budget)[^\d]*([\d,]+(?:\.\d+)?)\s*k?\b",
        r"\b([\d,]+)\s*dollars?\b",
        r"\b(\d{4,6})\b",   # fallback: bare 4-6 digit number
    ]
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            # Handle "20k" → "20000"
            if m.group(0).lower().endswith("k"):
                raw = str(int(float(raw) * 1000))
            return raw
    return None


def _extract_year(text: str) -> Optional[str]:
    """Try to extract a 4-digit model year from a natural-language prompt."""
    # Match things like: "2018 or newer", "from 2016", "minimum year 2019"
    patterns = [
        r"(?:from|since|after|min(?:imum)?\s+year|year\s+(?:from|min(?:imum)?)?)[^\d]*(20\d{2})",
        r"(20\d{2})\s+(?:or\s+newer|onward|onwards|and\s+up)",
        r"\b(20\d{2})\b",  # bare year fallback
    ]
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            return m.group(1)
    return None
