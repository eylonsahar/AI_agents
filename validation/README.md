# Validation Suite — AItzik Pre-Deployment Testing

Automated test suite for validating the AItzik AI Car Agent before deployment. Designed to catch bugs, verify graded requirements, and probe the intent classifier with adversarial prompts.

---

## Design

### Tiered Architecture

Tests are organized into four tiers by cost and dependency:

| Tier | Name | Server | LLM | Purpose |
|------|------|--------|-----|---------|
| **0** | Unit tests | No | No | Pure Python: `_extract_price`, `_extract_year` regex logic |
| **1** | API contract | Yes | No | GET endpoints, `agent_info` completeness, frontend smoke, input validation |
| **2** | Intent classifier | Yes | Yes | Adversarial prompts — expect rejection of non-car requests |
| **3** | Full pipeline | Yes | Yes | End-to-end runs, schema validation, efficiency tracking |

You can run any combination of tiers via `--tiers 0 1 2 3`.

### Prompt Generation Agents

Two agents generate prompts for Tier 2 and Tier 3 when `--random` is used. Both inherit from an abstract `UserAgent` base:

```
UserAgent (abstract)
├── _get_llm()      — lazy LLM gateway init
├── _call_llm()     — single batch LLM call
├── _parse_lines()  — parse output into lines
└── generate(n)      — abstract, returns list of prompts

GoodUserAgent       — valid car-search prompts (Tier 3)
BadUserAgent        — adversarial non-car prompts (Tier 2)
```

The **BadUserAgent** acts as an adversarial generator: it tries to fool AItzik’s classifier with prompts that look car-related but are not (flights, real estate, motorcycles, etc.). Prompts that slip through become high-value bug reports. BadUserAgent splits prompts into three categories (off-topic, car-adjacent, ambiguous), so when using `--random N` with Tier 2, **N must be divisible by 3**. The CLI validates this before starting any heavy phases and exits with a clear message if not.

### Save / Replay

When `--random` is used, generated prompts are saved to `validation/reports/saved_prompts_<timestamp>.json`. Use `--replay <file>` to re-run the same prompts after fixes, for deterministic regression testing.

### Budget Guard

Tier 2 and Tier 3 consume LLM budget. The suite limits pipeline runs based on `--budget` (default 3.0 USD). When the limit is reached, remaining tests are skipped and marked `SKIPPED (budget exhausted)`.

### AItzik Response Snapshots

For Tier 2 and Tier 3 end-to-end runs, the suite captures AItzik’s final response text (up to 2000 chars). This is logged in the report under **AITZIK RESPONSE SNAPSHOTS** for manual review. It is not auto-tested but useful for team presentation and quality checks.

### Exception Handling

Each test is isolated: a failure in one test does not stop the suite. When an unexpected exception occurs (e.g. network error, bug in the code under test):

- The full traceback is captured
- The request that caused the failure is saved (`request_snapshot`) for replay
- Both are written to the report under **UNEXPECTED EXCEPTIONS (code failures)**

You can use `request_snapshot` to replay the failing request after fixing the bug.

---

## Directory Structure

```
validation/
├── README.md           # This file
├── __init__.py        # Package exports
├── models.py          # TestResult, ROOT, constants
├── base_user_agent.py # Abstract UserAgent
├── good_user_agent.py # GoodUserAgent
├── bad_user_agent.py  # BadUserAgent
├── tier0.py           # Unit tests (extract_price, extract_year)
├── tier1.py           # API contract tests
├── tier2.py           # Intent classifier tests
├── tier3.py           # Full pipeline tests
├── test_auto_suite.py  # AItzikTestSuite, CLI, report generation
├── samples/           # Sanitized sample for replay (git-tracked)
│   └── saved_prompts_sample.json
└── reports/           # Auto-created, gitignored: test_report_*.json, test_report_*.txt, saved_prompts_*.json
```

---

## How to Run

### Prerequisites

- Python environment with `requests` installed
- For Tier 1–3: API server running (e.g. `uvicorn api.server:app --port 8001`)
- For Tier 2–3 with `--random`: `OPENAI_API_KEY` in `.env`

### CLI

```bash
# Free tests only (Tier 0 + 1) — no cost
python -m validation.test_auto_suite --tiers 0 1 --url http://localhost:8001

# Tier 0 only (no server needed)
python -m validation.test_auto_suite --tiers 0

# Full suite with fixed prompts (no random generation)
python -m validation.test_auto_suite --tiers 0 1 2 3 --budget 3.0

# Full suite with LLM-generated random prompts (6 prompts per agent by default)
python -m validation.test_auto_suite --tiers 0 1 2 3 --random --budget 3.0

# Random mode with custom number of prompts (with Tier 2, N must be divisible by 3)
python -m validation.test_auto_suite --tiers 0 1 2 3 --random 9 --budget 3.0

# Replay a previous random run (deterministic regression)
python -m validation.test_auto_suite --tiers 2 3 --replay validation/samples/saved_prompts_sample.json --budget 3.0

# Against deployed Render instance
python -m validation.test_auto_suite --tiers 0 1 2 3 --url https://your-app.onrender.com --budget 2.0
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tiers` | `0 1` | Tiers to run (space-separated) |
| `--random [N]` | off | Use LLM to generate N prompts for Tier 2/3 (default 6). With Tier 2, N must be divisible by 3 |
| `--replay` | — | Load prompts from saved JSON file |
| `--budget` | 3.0 | Max USD for pipeline runs |
| `--url` | http://localhost:8001 | Base URL of the API server |

### Programmatic Usage

```python
from validation import AItzikTestSuite, GoodUserAgent, BadUserAgent

# Run suite
suite = AItzikTestSuite(base_url="http://localhost:8001", budget_limit_usd=3.0)
suite.run(tiers=[0, 1, 2, 3], random_mode=True, num_examples=6)
json_path, txt_path = suite.generate_report(tiers_run=[0, 1, 2, 3], random_mode=True, replay_file=None, num_examples=6)

# Use agents directly
good = GoodUserAgent()
prompts = good.generate(6)

bad = BadUserAgent()
adversarial = bad.generate(6)
```

---

## Report Output

After each run, two files are written to `validation/reports/`:

- **`test_report_<timestamp>.json`** — Machine-readable: summary, efficiency, module consistency, all test results (including `response_snapshot`, `exception_traceback`, `request_snapshot` per test)
- **`test_report_<timestamp>.txt`** — Human-readable: GRADED REQUIREMENTS, FAILED TESTS, AITZIK RESPONSE SNAPSHOTS, UNEXPECTED EXCEPTIONS, EFFICIENCY, ADVERSARIAL RESULTS, SKIPPED

When `--random` is used, **`saved_prompts_<timestamp>.json`** is also written for replay.
