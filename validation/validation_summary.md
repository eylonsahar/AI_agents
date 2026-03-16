# Validation Summary — AItzik AI Car Agent
**Report based on:** `test_report_20260311_015717`  
**Run:** Tiers 0,1,2,3 · Random 9 · 20 pipeline runs · 58 tests · **51 passed / 7 failed**

---

## Legend

| Severity | Meaning |
|----------|---------|
| 🔴 HIGH | Directly broken user-facing feature / wrong results returned |
| 🟠 MEDIUM | Partial failure, edge case, or inconsistent behavior |
| 🟡 LOW | Cosmetic, minor edge case, or potential future issue |

| Complexity | Meaning |
|------------|---------|
| ★★★ HARD | Requires deep pipeline or regex logic changes |
| ★★ MEDIUM | Localized fix, moderate effort |
| ★ EASY | Trivial one-liner or config change |

---

## Bug #1 — `_extract_price` misidentifies year as price when both appear without `$`

**Tier:** 0 (unit test failure)  
**Severity:** 🔴 HIGH  
**Complexity:** ★★ MEDIUM  

**Failing cases:**
- `"car from 1999 under 20000"` → expected `"20000"`, got `"1999"`
- `"20k budget from 2018"` → expected `"20000"`, got `"2018"`

**Root cause:** The regex for price extraction picks up the first 4-digit number it finds. When the user writes a bare year before a bare price (no `$` sign), the year wins. The `k`-suffix shorthand (`20k`) is also not normalized before the year-collision check.

**Impact:** If a user types `"car from 2019 under 15000"`, the pipeline will search for a max price of $2019 instead of $15,000, returning zero or wrong results.

**Fix direction:** Run year-extraction first and mask matched year tokens before running price extraction; or anchor the price regex to patterns that can't be years (i.e. only match if `$` prefix, `k` suffix, `,` separator, or word-boundary context like `under/max/budget`).

---

## Bug #2 — `_extract_year` returns `None` when year appears before price with no suffix

**Tier:** 0 (unit test failure)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  

**Failing case:**
- `"car from 1999 under 20000"` → expected `"1999"`, got `None`

**Root cause:** The same collision logic that misidentifies `1999` as the price also causes the year extractor to fail — it returns `None` because the number was already consumed or the pattern doesn't match in this ambiguous context.

**Impact:** User queries referencing very old cars (pre-2000 model years) will silently drop the year filter, potentially returning newer cars.

**Fix direction:** Coupled with Bug #1. Separate year and price extraction passes, with mutual exclusion on the matched token.

---

## Bug #3 — "No results" test passes when it shouldn't (`Toyota Okozaky` returns results)

**Tier:** 3 (fixed regression test failure)  
**Severity:** 🟠 MEDIUM 
**Complexity:** ★★ MEDIUM  
**HumaNote:** this might not be a bug... needs to discuss with eylon/aviv, maybe return the closest match is good, but still expect the model to anounce somthing like "no such car found, but heres asimilar result might interest you..."

**Test:** `"Toyota Okozaky under $50000, 2020 or newer"` — a deliberately nonsense car model. Expected `status=error`. Got `status=ok` with Toyota Yaris recommendations.

**What AItzik returned:**
> "🌟 AItzik's TOP RECOMMENDATIONS — TOYOTA YARIS — Meets the minimum year requirement (starts at 2020)..."

**Root cause:** The RAG/vector search is fuzzy enough that "Okozaky" (or the "Toyota" keyword alone) retrieves Toyota Yaris embeddings. The agent then proceeds with confidence. There is no validation step that checks whether the retrieved model actually matches the user's requested model name.

**Impact:** Users searching for nonexistent or misspelled models receive plausible-looking but wrong results with no indication that their requested model wasn't found.

**Suspicious observation (from responses):** The `tier3_spec_compliant_prompt_only` and `tier3_explicit_fields` both return Mitsubishi Outlander as top result but the prices quoted in the "why" text are in **£ (GBP)** while the listing prices are in **$ (USD)**. The RAG knowledge base may be mixing UK and US market data, causing the model reasoning to use UK price thresholds for US user queries.

---

## Bug #4 — Pipeline hangs / times out on specific prompt patterns

**Tier:** 3 (2 × hard timeout failures at 600s)  
**Severity:** 🔴 HIGH  
**Complexity:** ★★★ HARD  

**Prompts that timed out:**
1. `"Searching for a family SUV under $25,000 USD, minimum model year 2017."` — 2359 seconds (39 min), then timeout
2. `"Interested in a sporty coupe, max price $30,000 USD, min model year 2018."` — 8070 seconds (134 min!), then timeout

**Root cause:** Unknown. The pipeline entered an apparent infinite or very deep loop. The efficiency log for similar prompts shows 37–56 steps, so something about these specific queries caused the agent to loop. Candidates: agent retrying with no termination condition, recursive tool call, or LLM stuck in a planning loop.

**Impact:** Blocks the server entirely for 10–130 minutes, degrading all other requests. In production, this would be a complete availability incident.

**Fix direction:** Add a hard agent step limit (e.g. 60 steps max) with a forced `FinalAnswer` if exceeded; add a server-side request timeout (not just client-side).

---

## Bug #5 — Tier 2: Motorbikes, F-350, and RVs reach the pipeline instead of being rejected by the classifier

**Tier:** 2 (passed, but suspicious — marked as correctly rejected via "no vehicles found" not via intent classifier)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  

**Prompts and responses:**
- `"Looking for a 2014 Harley-Davidson Sportster around $7,900"` → `"No vehicles found matching: 'Looking for a 2014 Harley-Davidson Sportster'"`
- `"Need a 2019 Ford F-350 diesel crew cab for work, budget $28,000"` → `"No vehicles found matching: 'Need a 2019 Ford F-350 diesel crew cab for work'"` 
- `"Searching for a 2016 Class C RV 28ft listed at $34,500"` → `"No vehicles found matching: 'Searching for a 2016 Class C RV'"`

**Why this is fishy:** These three passed the test (`status != ok`), but they were rejected by the **search pipeline** finding no listings — not by the **intent classifier**. The classifier let them through as valid car searches. A motorcycle, a heavy-duty work truck, and an RV slipped through the intent gate.

**Impact:** These queries waste LLM tokens and pipeline steps (22–30 seconds each vs ~4 seconds for truly rejected prompts). If the database ever gains listings for motorcycles or RVs, they would be returned as valid results.

**Fix direction:** Extend the intent classifier prompt to explicitly reject: motorcycles, heavy-duty commercial trucks, RVs/motorhomes, boats, and car parts.

---

## Bug #6 — Year filter not enforced on returned listings (listings violate `min_year`)

**Tier:** 3 (passed, but wrong results)  
**Severity:** 🔴 HIGH  
**Complexity:** ★★ MEDIUM  

**Observed in multiple responses:**

| Prompt | Min year requested | Listing year returned |
|--------|--------------------|-----------------------|
| `"Need a used pickup truck, min year 2019"` | 2019 | **2011** (Ford Ranger) |
| `"Want a used minivan, model year 2016 or newer"` | 2016 | **2014** (Ford C-Max) |
| `"Searching for a small crossover SUV, minimum year 2012"` | 2012 | **2011** (Nissan Juke) |
| `"Reliable family SUV, 2018 or newer"` | 2018 | **2014** (Mitsubishi Outlander at $3,999) |

**Root cause:** The `ListingsRetriever` or `DecisionAgent` returns listings that violate the `year_min` constraint. The agent scores and presents them anyway rather than filtering them out.

**Impact:** Users are shown cars that don't match their stated minimum year requirement. This is a correctness failure — the core promise of the product is broken.

**Fix direction:** Apply a hard filter on `year >= year_min` in the `ListingsRetriever` or the `DecisionAgent` scoring step, before results reach the user.

---

## Bug #7 — Listing prices contain clearly invalid values ($0, $1, $189, $296)

**Tier:** 3 (passed, but suspicious)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★ EASY  

**Observed in responses:**

| Model | Price in response |
|-------|------------------|
| Hyundai Ioniq (2019, Sacramento) | **$0** |
| BMW 3 Series (2014, Daytona Beach) | **$1** |
| Nissan Juke (2011, Boise) | **$189** |
| BMW 3 Series (2017, Sacramento) | **$296** |

**Root cause:** Likely bad/test data in the listings database. The agent returns these listings without flagging them as suspicious.

**Impact:** Users see absurd prices. Even if the data is test data, the agent should validate and filter out listings with `price < 500` (or similar floor) before presenting them. 

**Fix direction:** Add a minimum-price sanity filter in the `ListingsRetriever` or `DecisionAgent` (e.g. drop any listing with `price < 200`); or flag them in the response as "price may be invalid".

**HumaNote:** should be a *field agent* responsabality, if price < TRESHOLD (say 500$) validate with seller for real price, treat this as missing data.

---

## Bug #8 — Luxury sedan query returns zero results (coverage gap)

**Tier:** 3 (failed with `status=error`)  
**Severity:** 🟡 LOW  
**Complexity:** ★★★ HARD  

**Prompt:** `"Hunting for a luxury sedan, max price $45,000 USD, minimum model year 2020."`  
**Response:** `"No vehicles found matching. Try broader keywords."`

**Root cause:** The RAG vector store does not contain luxury sedan models (BMW 5-series, Mercedes C-class, Audi A4, etc.) from 2020+. This is a data coverage gap, not a code bug.

**Impact:** A common, high-value query category returns zero results. Users looking for mainstream luxury sedans get nothing.

**Fix direction:** Expand the vehicle knowledge base to include common luxury sedan models. This is a data/RAG pipeline task, not a code fix.

**HumaNote:** just make sure manually no such car is available, if so this ok.
---

## Bug #9 — `tier3_explicit_fields` returns listings with year **below** `year_min=2018`

**Tier:** 3 (passed, but suspicious)  
**Severity:** 🔴 HIGH  
**Complexity:** ★★ MEDIUM  

**Prompt:** `{"prompt": "family SUV", "max_price": "22000", "year_min": "2018"}`  
**Top result returned:** Mitsubishi Outlander 2014 at $3,999

This is a duplicate of Bug #6 but occurring specifically on the `explicit_fields` path (where `year_min` is passed directly as a structured field rather than extracted from text). This confirms the filter is broken at the pipeline level regardless of how year is provided.

---

## Bug #10 — Currency mismatch: RAG knowledge base uses £ GBP, listings use $ USD

**Tier:** 3 (passed, but suspicious)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  

**Evidence:** In every "why this model" explanation, the agent reasons about prices in **£ (GBP)** (e.g., "£20,000–£22,000", "£7,000 for used Ioniq"), while:
- The user prompt is in **USD**
- The actual listings are in **USD**

**Root cause:** The RAG knowledge base was built from UK automotive sources (likely What Car? or Auto Express). The agent uses this information to reason about price fit, but the thresholds don't apply to the US market.

**Impact:** The "why this model" justification is based on wrong price thresholds. A car described as fitting the budget at "£20,000" is not the same as fitting a "$22,000" budget. This could cause:
- Models recommended as fitting the budget that are actually over-budget in the US market
- Models rejected as too expensive that are actually affordable in the US market

**Fix direction:** Rebuild RAG embeddings from US market sources, or add a currency conversion note in the reasoning prompt; or add a disclaimer in the response when RAG data references £ prices.

---

## Summary Table

| # | Bug | Severity | Complexity | Source | owner | notes
|---|-----|----------|------------|--------|
| 1 | `_extract_price` picks year instead of price (bare numbers) | 🔴 HIGH | ★★ MEDIUM | Tier 0 failure | Ron
| 2 | `_extract_year` returns None when year consumed by price regex | 🟠 MEDIUM | ★★ MEDIUM | Tier 0 failure | Ron
| 3 | Nonsense model name `Okozaky` returns Toyota Yaris results | 🔴 HIGH | ★★ MEDIUM | Tier 3 failure | Ron
| 4 | Pipeline hangs indefinitely on some prompts (>10 min timeout) | 🔴 HIGH | ★★★ HARD | Tier 3 failure | Eylon
| 5 | Motorbikes / work trucks / RVs bypass intent classifier | 🟠 MEDIUM | ★★ MEDIUM | Tier 2 suspicious pass | Ron | Not a car!
| 6 | Returned listings violate `min_year` constraint | 🔴 HIGH | ★★ MEDIUM | Tier 3 suspicious pass | Eylon
| 7 | Listings with prices $0, $1, $189, $296 returned to users | 🟠 MEDIUM | ★ EASY | Tier 3 suspicious pass | Ron
| 8 | Luxury sedan category has zero data coverage | 🟡 LOW | ★★★ HARD | Tier 3 failure | Eylon
| 9 | `explicit_fields` path also violates year filter | 🔴 HIGH | ★★ MEDIUM | Tier 3 suspicious pass | Eylon | like 6
| 10 | RAG uses £ GBP pricing; users/listings use $ USD | 🟠 MEDIUM | ★★ MEDIUM | Tier 3 suspicious pass | Ron |supervisor

---

## Recommended Fix Priority

1. **[Blocking]** Bug #6 / #9 — Year filter not enforced. Core product feature.
2. **[Blocking]** Bug #4 — Infinite loop / pipeline hang. Availability risk.
3. **[High]** Bug #3 — Nonsense model returns confident results.
4. **[High]** Bug #1 / #2 — Price/year regex collision.
5. **[Medium]** Bug #10 — Currency mismatch in RAG reasoning.
6. **[Medium]** Bug #5 — Weak intent classifier for motorcycles/RVs.
7. **[Low]** Bug #7 — Invalid listing prices ($0, $1).
8. **[Low]** Bug #8 — Data coverage gap for luxury sedans.
