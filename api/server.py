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

# ── Logging — all output goes to stdout so Render captures it ─────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
# Keep only noisy low-level libraries quiet
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
# Suppress the harmless OUTPUT_PARSING_FAILURE spam from LangChain parsers
logging.getLogger("langchain_core.output_parsers").setLevel(logging.ERROR)


from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a clean JSON error instead of Pydantic's raw 422 detail."""
    # Collect human-readable messages for each bad field
    messages = []
    for err in exc.errors():
        field = " -> ".join(str(x) for x in err.get("loc", []) if x != "body")
        msg   = err.get("msg", "invalid value")
        messages.append(f"'{field}': {msg}" if field else msg)
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error":  "Invalid request: " + "; ".join(messages),
            "response": None,
            "steps": [],
        },
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
        response_text, _ = llm_gateway.call_llm(
            prompt=(
                "You are a request classifier. "
                "Answer ONLY with the single word 'yes' or 'no'. "
                "'yes' = the request is about finding / buying a used car or vehicle. "
                "'no' = the request is about anything else.\n\n"
                f"Is this a car search request? Request: '{text}'"
            )
        )
        return "yes" in response_text.strip().lower()[:10]
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
            "your requirements, fills in missing listing data by contacting sellers, "
            "and schedules Google Calendar meeting links so you can view each car."
        ),
        "purpose": (
            "Help users find the best used vehicles on the market by automating: "
            "(1) semantic vehicle-model search via RAG + Pinecone, "
            "(2) live listings retrieval from a structured CSV database, "
            "(3) data enrichment via a Field Agent that interviews sellers, and "
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
                "prompt": "Reliable family SUV, prefer silver or white",
                "max_price": "22000",
                "year_min": "2018",
                "full_response": (
                    "🌟 AItzik's TOP RECOMMENDATIONS\n\n"
                    "━━ HYUNDAI SANTA FE ━━\n"
                    "💡 Why this model: Meets the user's year requirement (2018+), is explicitly described as a large, "
                    "practical seven-seat family SUV and is noted as good value for money. The retrieved data states "
                    "you'll need around £16,000 for a used Santa Fe, which is within the user's £22,000 max price. "
                    "The description emphasizes roomy second and third rows, flexible interior space and family-friendly "
                    "equipment, directly matching the request for a reliable family SUV.\n\n"
                    "  Option #1:\n"
                    "    • Region: ocala\n    • Price: 14500\n    • Year: 2018.0\n    • Condition: excellent\n"
                    "    • Paint Color: blue\n    • State: fl\n    • Accident: none\n    • Mileage: 84200\n"
                    "    📅 Schedule a viewing:\n"
                    "       🔗 2026-03-05 18:15: https://www.google.com/calendar/render?action=TEMPLATE&text=View+Car%3A+hyundai+santa+fe&dates=20260305T181500%2F20260305T184500&details=Car+ID%3A+7315970890&sf=true\n"
                    "       🔗 2026-03-08 11:30: https://www.google.com/calendar/render?action=TEMPLATE&text=View+Car%3A+hyundai+santa+fe&dates=20260308T113000%2F20260308T120000&details=Car+ID%3A+7315970890&sf=true\n\n"
                    "  Option #2:\n"
                    "    • Region: inland empire\n    • Price: 19900\n    • Year: 2019.0\n    • Condition: excellent\n"
                    "    • Paint Color: red\n    • State: ca\n    • Accident: 2021-08-14\n    • Mileage: 84250\n"
                    "    📅 Schedule a viewing:\n"
                    "       🔗 2026-03-04 18:00: https://www.google.com/calendar/render?action=TEMPLATE&text=View+Car%3A+hyundai+santa+fe&dates=20260304T180000%2F20260304T183000&details=Car+ID%3A+7316590169&sf=true\n"
                    "       🔗 2026-03-08 11:30: https://www.google.com/calendar/render?action=TEMPLATE&text=View+Car%3A+hyundai+santa+fe&dates=20260308T113000%2F20260308T120000&details=Car+ID%3A+7316590169&sf=true\n\n"
                    "━━ MITSUBISHI OUTLANDER ━━\n"
                    "💡 Why this model: Includes model years within the user's minimum (2018-2021 examples included) "
                    "and pricing ranges show many examples available at or below £22,000. The Outlander is described "
                    "as a good-value, well-equipped large SUV with family appeal and a five-star Euro NCAP rating.\n\n"
                    "  Option #3:\n"
                    "    • Region: columbus\n    • Price: 18997\n    • Year: 2019.0\n    • Condition: excellent\n"
                    "    • Paint Color: grey\n    • State: oh\n    • Accident: none\n    • Mileage: 85200\n"
                    "    📅 Schedule a viewing:\n"
                    "       🔗 2026-03-06 18:00: https://www.google.com/calendar/render?action=TEMPLATE&text=View+Car%3A+mitsubishi+outlander&dates=20260306T180000%2F20260306T183000&details=Car+ID%3A+7317007116&sf=true\n"
                    "       🔗 2026-03-11 10:30: https://www.google.com/calendar/render?action=TEMPLATE&text=View+Car%3A+mitsubishi+outlander&dates=20260311T103000%2F20260311T110000&details=Car+ID%3A+7317007116&sf=true\n\n"
                    "Click the links above to add viewings to your calendar!"
                ),
                "steps": [
                    {"module": "Supervisor / Thought", "prompt": "", "response": "I'll search the database for vehicle models that match the user's requirements (max $22,000, min year 2018, reliable family SUVs, prefer silver or white)."},
                    {"module": "SearchPipeline / VehicleModelRetriever", "prompt": "max price: 22000 min year: 2018 Reliable family SUV, prefer silver or white", "response": "Hyundai Santa Fe (4X4, 2018-2024) | match: 0.95 | Meets the user's year requirement (2018+), is explicitly described as a large, practical seven-seat family SUV and is noted as good value for money.\nMitsubishi Outlander (4X4, 2013-2021) | match: 0.86 | Good-value, well-equipped large SUV with family appeal, five-star Euro NCAP rating, PHEV variant offers very low running costs."},
                    {"module": "Supervisor / Thought", "prompt": "", "response": "I'll retrieve for-sale listings for the found models (Hyundai Santa Fe and Mitsubishi Outlander) that match the user's constraints."},
                    {"module": "SearchPipeline / ListingsRetriever", "prompt": "Hyundai Santa Fe, Mitsubishi Outlander", "response": "Total: 6 listings | Hyundai Santa Fe: 3 listings, Mitsubishi Outlander: 3 listings"},
                    {"module": "SearchPipeline / DecisionAgent", "prompt": "Score and rank 6 listings", "response": "#1: hyundai santa fe (2019.0, $19900) | score: 0.85\n#2: hyundai santa fe (2018.0, $14500) | score: 0.78\n#3: mitsubishi outlander (2019.0, $18997) | score: 0.71\n#4: mitsubishi outlander (2014.0, $3999) | score: 0.71"},
                    {"module": "Supervisor / Thought", "prompt": "", "response": "I'll delegate completing the 6 retrieved listings and scheduling meetings with sellers."},
                    {"module": "FieldAgent / Thought", "prompt": "", "response": "This listing is missing accident and mileage; request both fields from the seller for listing 7316590169 before scheduling a meeting."},
                    {"module": "FieldAgent / Seller/GetData", "prompt": "Requested fields for listing 7316590169: accident, mileage", "response": "mileage = 84250\naccident = 2021-08-14\npaint_color = silver\nstate = CA\nid = 7316590169"},
                    {"module": "FieldAgent / Thought", "prompt": "", "response": "All required fields for listing 7316590169 were just filled; schedule a meeting now."},
                    {"module": "FieldAgent / Seller/Scheduling", "prompt": "Requested 2 available slots from 2026-03-04 to 2026-03-18", "response": "2026-03-04 18:00\n2026-03-08 11:30"},
                    {"module": "FieldAgent / Thought", "prompt": "", "response": "This listing is missing accident and mileage; request both fields from the seller for listing 7315970890 before scheduling a meeting."},
                    {"module": "FieldAgent / Seller/GetData", "prompt": "Requested fields for listing 7315970890: accident, mileage", "response": "mileage = 84200\naccident = none\npaint_color = silver\nstate = CA\nid = 7315970890"},
                    {"module": "FieldAgent / Thought", "prompt": "", "response": "All required fields for listing 7315970890 are filled; schedule a meeting now."},
                    {"module": "FieldAgent / Seller/Scheduling", "prompt": "Requested 2 available slots from 2026-03-04 to 2026-03-18", "response": "2026-03-05 18:15\n2026-03-08 11:30"},
                    {"module": "FieldAgent / Seller/GetData", "prompt": "Requested fields for listing 7315719756: accident, mileage", "response": "mileage = 84250\naccident = none\npaint_color = silver\nstate = CA\nid = 7315719756"},
                    {"module": "FieldAgent / Seller/Scheduling", "prompt": "Requested 2 available slots from 2026-03-04 to 2026-03-18", "response": "2026-03-04 17:30\n2026-03-08 11:00"},
                    {"module": "FieldAgent / Seller/GetData", "prompt": "Requested fields for listing 7317007116: accident, mileage", "response": "mileage = 85200\naccident = none\npaint_color = silver\nstate = CA\nid = 7317007116"},
                    {"module": "FieldAgent / Seller/Scheduling", "prompt": "Requested 2 available slots from 2026-03-04 to 2026-03-18", "response": "2026-03-06 18:00\n2026-03-11 10:30"},
                    {"module": "FieldAgent / Seller/GetData", "prompt": "Requested fields for listing 7316941545: accident, mileage", "response": "mileage = 94200\naccident = none\npaint_color = silver\nstate = ca\nid = 7316941545"},
                    {"module": "FieldAgent / Seller/Scheduling", "prompt": "Requested 2 available slots from 2026-03-04 to 2026-03-18", "response": "2026-03-05 18:00\n2026-03-08 11:15"},
                    {"module": "FieldAgent / Seller/GetData", "prompt": "Requested fields for listing 7316872707: accident, mileage", "response": "mileage = 84250\naccident = 2021-08-15\npaint_color = silver\nstate = CA\nid = 7316872707"},
                    {"module": "FieldAgent / Seller/Scheduling", "prompt": "Requested 2 available slots from 2026-03-04 to 2026-03-18", "response": "2026-03-08 11:00\n2026-03-11 18:30"},
                    {"module": "FieldAgent / FinalAnswer", "prompt": "", "response": "Completed processing for 6 listings. Missing fields filled and meetings scheduled (2 slots each) for listings: 7316590169, 7315970890, 7315719756, 7317007116, 7316941545, 7316872707."},
                    {"module": "SearchPipeline / DecisionAgent (final re-ranking)", "prompt": "Re-rank 6 fully enriched listings", "response": "#1: hyundai santa fe (2018.0, $14500) | score: 0.79\n#2: hyundai santa fe (2019.0, $19900) | score: 0.78\n#3: mitsubishi outlander (2019.0, $18997) | score: 0.71\n#4: mitsubishi outlander (2014.0, $8995) | score: 0.69\n#5: mitsubishi outlander (2014.0, $3999) | score: 0.63\n#6: hyundai santa fe (2018.0, $27675) | score: 0.51"},
                    {"module": "Supervisor / FinalAnswer", "prompt": "", "response": "Mission complete."}
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

@app.post("/api/execute")
def execute(body: ExecuteRequest, stream: bool = False):
    """
    Run the full multi-agent pipeline.

    ?stream=true  → NDJSON: processing line, heartbeat(s), then final result  (used by UI)
    ?stream=false → single blocking JSON response                              (default, for curl)
    """
    def _stream():
        import json as _json

        # ── Fast validation ───────────────────────────────────────────────
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            yield _json.dumps({"status": "error", "error": "OPENAI_API_KEY not configured.",
                                "response": None, "steps": []}) + "\n"
            return

        prompt_text = (body.prompt or "").strip()
        if not prompt_text:
            yield _json.dumps({"status": "error", "error": "Prompt cannot be empty.",
                                "response": None, "steps": []}) + "\n"
            return

        max_price   = body.max_price   or _extract_price(prompt_text)
        year_min    = body.year_min    or _extract_year(prompt_text)
        description = body.description or prompt_text

        if not max_price:
            yield _json.dumps({"status": "error",
                                "error": ("Could not determine your maximum price. "
                                          "Please include a budget, e.g. 'max price $20000', "
                                          "or use the max_price field."),
                                "response": None, "steps": []}) + "\n"
            return

        if not year_min:
            yield _json.dumps({"status": "error",
                                "error": ("Could not determine a minimum year. "
                                          "Please include a year, e.g. 'from 2018 onwards', "
                                          "or use the year_min field."),
                                "response": None, "steps": []}) + "\n"
            return

        # ── Intent check (fast LLM call) ──────────────────────────────────
        llm_gateway = LLMGateway.get_instance(api_key=api_key)
        if not _check_is_car_search(prompt_text, description, llm_gateway):
            yield _json.dumps({"status": "error",
                                "error": ("Your request doesn't seem to be about finding a car. "
                                          "Please describe the vehicle you're looking for, e.g. "
                                          "'reliable family SUV under $20,000 from 2018 or newer'."),
                                "response": None, "steps": []}) + "\n"
            return

        # ── Emit acknowledgement immediately ──────────────────────────────
        yield _json.dumps({
            "status": "processing",
            "message": (
                "✅ Request received! Your AI car agent is now searching for the best "
                "listings, contacting sellers, and scheduling viewings. "
                "This usually takes 10 minutes — we'll return the full results once ready."
            )
        }) + "\n"

        # ── Run agent in background thread + send heartbeats every 30s ──
        import threading as _threading

        result_holder: dict = {}

        def _run():
            try:
                requirements = {"max_price": max_price, "year_min": year_min,
                                 "description": description}
                sup = AgentSupervisor(
                    llm_gateway=llm_gateway,
                    vehicle_retriever=_vehicle_retriever,
                )
                result_holder["result"] = sup.run_headless(requirements=requirements)
            except Exception as exc:
                traceback.print_exc()
                result_holder["error"] = str(exc)

        thread = _threading.Thread(target=_run, daemon=True)
        thread.start()

        # Heartbeat every 30 s keeps the TCP connection alive while we wait
        while thread.is_alive():
            thread.join(timeout=30)
            if thread.is_alive():
                yield _json.dumps({"status": "heartbeat"}) + "\n"

        # ── Agent finished — emit result ───────────────────────────────
        if "error" in result_holder:
            yield _json.dumps({"status": "error", "error": result_holder["error"],
                                "response": None, "steps": []}) + "\n"
            return

        result = result_holder.get("result", {})
        if result.get("error") == "no_vehicles_found":
            query = description or prompt_text
            yield _json.dumps({
                "status": "error",
                "error": (f"I couldn't find any vehicles matching: '{query}'. "
                          "Try different keywords — e.g. 'SUV', 'sedan', 'pickup truck', "
                          "or specific makes like 'Toyota', 'Honda', 'Ford'."),
                "response": None,
                "steps": _normalize_steps(result.get("steps", [])),
            }) + "\n"
            return

        yield _json.dumps({
            "status":   "ok",
            "error":    None,
            "response": _format_results_as_text(result.get("results", [])),
            "steps":    _normalize_steps(result.get("steps", [])),
        }) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson") \
        if stream else JSONResponse(_run_blocking(
            body, _extract_price, _extract_year,
            LLMGateway, _check_is_car_search, AgentSupervisor,
            _vehicle_retriever, _normalize_steps, _format_results_as_text, traceback,
        ))


def _run_blocking(body, _extract_price, _extract_year,
                   LLMGateway, _check_is_car_search, AgentSupervisor,
                   _vehicle_retriever, _normalize_steps, _format_results_as_text, traceback):
    """
    Blocking pipeline for the plain-curl path (no streaming).
    Accepts {"prompt": "..."} only — all other fields are extracted automatically.
    Returns a plain dict that JSONResponse will serialize.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"status": "error",
                "error": "Server configuration error: OPENAI_API_KEY is not set.",
                "response": None, "steps": []}

    # ── Validate prompt ───────────────────────────────────────────────────
    prompt_text = (body.prompt or "").strip()
    if not prompt_text:
        return {"status": "error",
                "error": "Missing required field: 'prompt' cannot be empty.",
                "response": None, "steps": []}

    if len(prompt_text) < 10:
        return {"status": "error",
                "error": "Prompt is too short. Please describe the car you're looking for in more detail.",
                "response": None, "steps": []}

    # ── Extract price and year from prompt ────────────────────────────────
    max_price = body.max_price or _extract_price(prompt_text)
    year_min  = body.year_min  or _extract_year(prompt_text)
    description = body.description or prompt_text

    if not max_price:
        return {"status": "error",
                "error": ("Could not extract a maximum budget from your prompt. "
                          "Please include a price, e.g. 'under $20,000' or 'max budget 15000'."),
                "response": None, "steps": []}

    if not year_min:
        return {"status": "error",
                "error": ("Could not extract a minimum model year from your prompt. "
                          "Please include a year, e.g. '2018 or newer' or 'from 2016'."),
                "response": None, "steps": []}

    # ── Intent validation ─────────────────────────────────────────────────
    llm_gateway = LLMGateway.get_instance(api_key=api_key)
    if not _check_is_car_search(prompt_text, description, llm_gateway):
        return {"status": "error",
                "error": ("Invalid request: the prompt does not appear to be a car search. "
                          "Please describe the vehicle you are looking for, e.g. "
                          "'reliable family SUV under $20,000 from 2018 or newer'."),
                "response": None, "steps": []}

    # ── Run the agent ─────────────────────────────────────────────────────
    try:
        sup    = AgentSupervisor(llm_gateway=llm_gateway, vehicle_retriever=_vehicle_retriever)
        result = sup.run_headless(requirements={"max_price": max_price,
                                                "year_min":  year_min,
                                                "description": description})
        if result.get("error") == "no_vehicles_found":
            return {"status": "error",
                    "error": (f"No vehicles found matching: '{description}'. "
                              "Try broader keywords, e.g. 'SUV', 'sedan', or a specific make."),
                    "response": None,
                    "steps": _normalize_steps(result.get("steps", []))}
        return {
            "status":   "ok",
            "error":    None,
            "response": _format_results_as_text(result.get("results", [])),
            "steps":    _normalize_steps(result.get("steps", []))
        }
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "error": str(exc), "response": None, "steps": []}



# ===========================================================================
# Extraction helpers
# ===========================================================================

import re as _re


def _extract_price(text: str) -> Optional[str]:
    """Try to extract a dollar amount from a natural-language prompt."""
    # Mask only the year in its matched context (not all occurrences) to avoid:
    # - "car from 1999 under 20000" → 20000 not 1999
    # - "under 2000 corola from 2000" → 2000 (don't mask the price occurrence)
    _year_patterns = [
        r"(?:from|since|after|min(?:imum)?\s+year|year\s+(?:from|min(?:imum)?)?)[^\d]*(19[89]\d|20\d{2})",
        r"(19[89]\d|20\d{2})\s+(?:or\s+newer|onward|onwards|and\s+up)",
        r"\b(19[89]\d|20\d{2})\b",
    ]
    for pat in _year_patterns:
        m = _re.search(pat, text, _re.IGNORECASE)
        if m:
            text = text[: m.start(1)] + "____YEAR____" + text[m.end(1) :]
            break

    # Patterns like: $20000, 20k, 1.5k, under 2000, under 2,500, max price 20000, budget 15000
    patterns = [
        r"\$\s*([\d,]+(?:\.\d+)?)\s*k?\b",
        r"\b([\d,]+(?:\.\d+)?)\s*k\b",  # 20k, 1.5k — decimal k-suffix supported
        r"(?:under|below)\s+([\d,]+(?:\.\d+)?)\s*k?\b",  # under 2000, under 2,500 (not max—keeps 2019 excluded)
        r"(?:max(?:imum)?\s+(?:price|budget)|budget)[^\d]*([\d,]+(?:\.\d+)?)\s*k?\b",
        r"\b([\d,]+)\s*dollars?\b",
        r"\b(?!19[89]\d\b)(?!20\d{2}\b)(\d{4,6})\b",  # bare 4-6 digit, NOT year (1980-2099)
    ]
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            # Handle "20k" / "1.5k" → "20000" / "1500"
            if m.group(0).lower().endswith("k"):
                raw = str(int(float(raw) * 1000))
            # under/below: reject 2001-2099 (year-like); allow 2000, 2500, 15000, etc.
            if "under" in pattern or "below" in pattern:
                n = int(float(raw))
                if 2001 <= n <= 2099:
                    continue
            return raw
    return None


def _extract_year(text: str) -> Optional[str]:
    """Try to extract a 4-digit model year from a natural-language prompt."""
    # Match things like: "2018 or newer", "from 1999", "minimum year 2019"
    # Support both 19xx (1980-1999) and 20xx (2000-2099) for model years
    patterns = [
        r"(?:from|since|after|min(?:imum)?\s+year|year\s+(?:from|min(?:imum)?)?)[^\d]*(19[89]\d|20\d{2})",
        r"(19[89]\d|20\d{2})\s+(?:or\s+newer|onward|onwards|and\s+up)",
        r"\b(19[89]\d|20\d{2})\b",  # bare year fallback (1980-2099)
    ]
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            return m.group(1)
    return None
