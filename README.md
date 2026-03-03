# 🚗 AItzik — AI Car Agent

AItzik is a multi-agent system that autonomously finds used vehicles matching your requirements, enriches missing listing data by contacting sellers, and schedules Google Calendar viewing links.

**Team:** Eylon Sahar, Aviv Rabi, Ron Bartal · Technion — Group 1_3

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  Supervisor Agent                    │
│  (LangChain ReAct · orchestrates the full pipeline) │
└───┬──────────────────────────────┬──────────────────┘
    │                              │
    ▼                              ▼
┌───────────────────┐   ┌──────────────────────────────┐
│   Search Pipeline │   │        Field Agent            │
│                   │   │  (LangChain ReAct · per car)  │
│ 1. RAG / Pinecone │   │                               │
│    Vehicle Model  │   │  • Contacts seller to fill    │
│    Retriever      │   │    missing fields (mileage,   │
│                   │   │    accident, color, etc.)     │
│ 2. CSV Listings   │   │                               │
│    Retriever      │   │  • Schedules 2 Google Calendar│
│                   │   │    viewing slots per listing  │
│ 3. Decision Agent │   │                               │
│    (score & rank) │   └──────────────────────────────┘
└───────────────────┘
```

---

## Project Structure

```
AI_agents/
├── api/
│   └── server.py              # FastAPI server — all HTTP endpoints
├── agents/
│   ├── supervisor_agent/
│   │   └── supervisor_agent.py  # Orchestrator (LangChain ReAct)
│   ├── field_agent/
│   │   ├── field_agent.py       # Per-listing enrichment agent
│   │   └── tools.py             # Seller mock + scheduling tools
│   ├── search_agent/
│   │   ├── vehicle_model_retriever.py  # RAG + Pinecone search
│   │   ├── rag_retrieval.py            # Generic RAG wrapper
│   │   └── listings_retriever.py      # CSV listings DB search
│   ├── action_log.py            # Shared execution trace logger
│   └── prompts.py               # All LLM system prompts
├── gateways/
│   ├── llm_gateway.py           # Singleton OpenAI LLM client
│   └── embedding_gateway.py     # Singleton OpenAI embeddings client
├── frontend/
│   └── index.html               # Chat UI (vanilla HTML/JS)
├── rag/                         # RAG indexing scripts
├── config.py                    # Shared constants & configuration
├── requirements.txt
├── test_api.sh                  # Shell-based API test suite
└── model_architecture.png       # System diagram (served via API)
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/eylonsahar/AI_agents.git
cd AI_agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=filtered-vehicles-info
```

### 3. Start the server

```bash
source .venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8001
```

The **web UI** is served at: [http://localhost:8001](http://localhost:8001)

---

## API Reference

### `GET /api/team_info`
Returns team name and student details.

```bash
curl http://localhost:8001/api/team_info
```

---

### `GET /api/agent_info`
Returns agent description, prompt template, and a real example with full steps.

```bash
curl http://localhost:8001/api/agent_info
```

---

### `GET /api/model_architecture`
Returns the system architecture diagram as a PNG image.

```bash
curl -o architecture.png http://localhost:8001/api/model_architecture
```

---

### `POST /api/execute`
**Main endpoint.** Runs the full multi-agent pipeline.

**Request body:**
```json
{
  "prompt": "Reliable family SUV under $22,000, 2018 or newer"
}
```

The `prompt` field is the only required field. The budget and minimum year are extracted automatically from the text. If they cannot be extracted, a descriptive error is returned.

**Success response:**
```json
{
  "status": "ok",
  "error": null,
  "response": "🌟 AItzik's TOP RECOMMENDATIONS\n\n━━ HYUNDAI SANTA FE ━━\n...",
  "steps": [
    { "module": "Supervisor / Thought",       "prompt": "", "response": "..." },
    { "module": "SearchPipeline / VehicleModelRetriever", "prompt": "...", "response": "..." },
    { "module": "FieldAgent / Seller/GetData", "prompt": "...", "response": "..." },
    { "module": "FieldAgent / Seller/Scheduling", "prompt": "...", "response": "..." }
  ]
}
```

**Error response** (invalid/incomplete prompt):
```json
{
  "status": "error",
  "error": "Could not extract a maximum budget from your prompt. Please include a price, e.g. 'under $20,000'.",
  "response": null,
  "steps": []
}
```

**curl example** (blocking, waits up to 10 minutes):
```bash
curl -s -X POST http://localhost:8001/api/execute \
  -H "Content-Type: application/json" \
  --max-time 600 \
  -d '{"prompt": "Reliable family SUV under $22,000, 2018 or newer, prefer silver"}'
```

> **Note:** The pipeline typically takes **2–5 minutes** to complete.  
> Use `--max-time 600` to avoid curl timing out early.

---

## Validation Rules

| Condition | Error |
|-----------|-------|
| `prompt` field missing | `"Missing required field: 'prompt' cannot be empty."` |
| Prompt shorter than 10 characters | `"Prompt is too short..."` |
| No budget found in prompt | `"Could not extract a maximum budget..."` |
| No year found in prompt | `"Could not extract a minimum model year..."` |
| Not a car-related request | `"Invalid request: the prompt does not appear to be a car search."` |
| No listings found | `"No vehicles found matching: '...'"` |

---

## Running Tests

```bash
# Full API test suite (requires running server on port 8001)
bash test_api.sh 8001

# Supervisor agent unit test
source .venv/bin/activate
python test_supervisor.py

# Field agent autonomous test
python test_field_agent_autonomous.py
```

---

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Agent framework | LangChain `create_react_agent` + `AgentExecutor` |
| LLM | OpenAI GPT-4o (via `LLMGateway`) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector DB | Pinecone |
| API server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript |
