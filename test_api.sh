#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AItzik API smoke tests
# Usage:  bash test_api.sh [port]
# Example: bash test_api.sh 8001
# ─────────────────────────────────────────────────────────────────────────────

PORT="${1:-8001}"
BASE="http://localhost:${PORT}"

GREEN='\033[0;32m'; RED='\033[0;31m'; RESET='\033[0m'
PASS=0; FAIL=0

check() {
  local label="$1" expected_status="$2" actual_status="$3"
  if [ "$actual_status" -eq "$expected_status" ]; then
    echo -e "${GREEN}✅ PASS${RESET}  $label (HTTP $actual_status)"
    PASS=$((PASS+1))
  else
    echo -e "${RED}❌ FAIL${RESET}  $label (expected $expected_status, got $actual_status)"
    FAIL=$((FAIL+1))
  fi
}

echo "=================================="
echo " AItzik API Tests — ${BASE}"
echo "=================================="
echo ""

# ── 1. GET /api/team_info ─────────────────────────────────────────────────
echo "▶  GET /api/team_info"
STATUS=$(curl -s -o /tmp/team_info.json -w "%{http_code}" "${BASE}/api/team_info")
check "/api/team_info" 200 "$STATUS"
if [ "$STATUS" -eq 200 ]; then
  TEAM=$(python3 -c "import json; d=json.load(open('/tmp/team_info.json')); print(d.get('team_name','?'), '|', len(d.get('students',[])),'students')" 2>/dev/null)
  echo "   team: $TEAM"
fi
echo ""

# ── 2. GET /api/agent_info ────────────────────────────────────────────────
echo "▶  GET /api/agent_info"
STATUS=$(curl -s -o /tmp/agent_info.json -w "%{http_code}" "${BASE}/api/agent_info")
check "/api/agent_info" 200 "$STATUS"
if [ "$STATUS" -eq 200 ]; then
  KEYS=$(python3 -c "import json; d=json.load(open('/tmp/agent_info.json')); print(', '.join(d.keys()))" 2>/dev/null)
  echo "   keys: $KEYS"
fi
echo ""

# ── 3. GET /api/model_architecture ───────────────────────────────────────
echo "▶  GET /api/model_architecture"
STATUS=$(curl -s -o /tmp/architecture.png -w "%{http_code}" "${BASE}/api/model_architecture")
check "/api/model_architecture" 200 "$STATUS"
if [ "$STATUS" -eq 200 ]; then
  BYTES=$(wc -c < /tmp/architecture.png)
  echo "   size: ${BYTES} bytes"
fi
echo ""

# ── 4. POST /api/execute — intent rejection (not a car search) ────────────
echo "▶  POST /api/execute  [intent check — should reject 'buy pizza']"
STATUS=$(curl -s -o /tmp/execute_reject.json -w "%{http_code}" \
  -X POST "${BASE}/api/execute" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"buy me a pizza","max_price":"100","year_min":"2020","description":"buy me a pizza"}')
check "/api/execute (intent reject)" 200 "$STATUS"
if [ "$STATUS" -eq 200 ]; then
  RESP_STATUS=$(python3 -c "import json; d=json.load(open('/tmp/execute_reject.json')); print(d.get('status'))" 2>/dev/null)
  echo "   response status: $RESP_STATUS  (expected: error)"
  [ "$RESP_STATUS" = "error" ] && echo -e "   ${GREEN}✅ correctly rejected non-car request${RESET}" || echo -e "   ${RED}❌ non-car request was not rejected${RESET}"
fi
echo ""

# ── 5. POST /api/execute — full pipeline (real OpenAI + Pinecone call) ────
echo "▶  POST /api/execute  [full pipeline — family SUV ~2 min]"
echo "   (this calls OpenAI + Pinecone — may take ~2 minutes)"
STATUS=$(curl -s -o /tmp/execute_full.json -w "%{http_code}" \
  --max-time 300 \
  -X POST "${BASE}/api/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Reliable family SUV, prefer silver or white",
    "max_price": "22000",
    "year_min": "2018",
    "description": "Reliable family SUV, prefer silver or white"
  }')
check "/api/execute (full pipeline)" 200 "$STATUS"
if [ "$STATUS" -eq 200 ]; then
  python3 - <<'EOF'
import json, sys
try:
    d = json.load(open('/tmp/execute_full.json'))
    print(f"   status   : {d.get('status')}")
    steps = d.get('steps', [])
    print(f"   steps    : {len(steps)}")
    resp = d.get('response') or ''
    print(f"   response : {resp[:120]}{'...' if len(resp)>120 else ''}")
    err = d.get('error')
    if err:
        print(f"   error    : {err}")
except Exception as e:
    print(f"   parse error: {e}")
EOF
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "=================================="
echo -e " Results: ${GREEN}${PASS} passed${RESET}  ${RED}${FAIL} failed${RESET}"
echo "=================================="
