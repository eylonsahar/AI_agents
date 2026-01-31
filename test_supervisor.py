from dotenv import load_dotenv
import os
from agents.supervisor_agent.supervisor_agent import AgentSupervisor
from gateways.llm_gateway import LLMGateway

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Path to the JSON file produced by the RAG agent (or hand-crafted for testing).
# Shape: {"vehicles": [{"make": "...", "model": "...", ...}, ...]}
RAG_RESULT_JSON = os.path.join(os.path.dirname(__file__), "agents/search_agents/test_result2_suv.json")

if __name__ == '__main__':
    llm = LLMGateway(api_key=OPENAI_API_KEY)

    supervisor = AgentSupervisor(
        llm_gateway=llm,
        rag_result_json_path=RAG_RESULT_JSON
    )
    output = supervisor.run()

    # ── sanity checks on the return value ──
    required_top_keys = {"results", "stats", "steps"}
    assert required_top_keys <= output.keys(), f"run() missing top-level keys: {required_top_keys - output.keys()}"
    print("✅ run() returned all required top-level keys (results, stats, steps)")

    # Every step must have the assignment-required schema
    required_step_keys = {"module", "submodule", "prompt", "response"}
    for i, step in enumerate(output["steps"]):
        missing = required_step_keys - step.keys()
        assert not missing, f"Step {i} missing keys: {missing}"
    print(f"✅ All {len(output['steps'])} steps have correct schema")

    # Both modules should appear — Supervisor logs its own decisions,
    # FieldAgent appends its decisions + MockSeller calls to the same log.
    modules_seen = {s["module"] for s in output["steps"]}
    assert "Supervisor" in modules_seen,  "No Supervisor steps in log"
    assert "FieldAgent" in modules_seen, "No FieldAgent steps in log"
    print("✅ Both Supervisor and FieldAgent steps present in shared log")

    print(f"\n📊 Stats: {output['stats']}")
