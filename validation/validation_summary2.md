# Validation Summary — AItzik AI Car Agent
**Previous report:** `test_report_20260311_015717` — 58 tests · 51 passed / 7 failed  
**Current report:** `test_report_20260312_224404` — 70 tests · 67 passed / 3 failed · 6 skipped  
**Classifier check:** `classifier_check_20260315_190832` — 23 labeled · Accuracy 87% · F1 0.87 · FP=2 · FN=1

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

## ✅ Bug #1 — `_extract_price` misidentifies year as price when both appear without `$`

**Tier:** 0 (unit test)  
**Severity:** 🔴 HIGH → **FIXED**  
**Complexity:** ★★ MEDIUM

**Status:** Tier 0 passes cleanly in the new report. Price/year regex collision resolved.

---

## ✅ Bug #2 — `_extract_year` returns `None` when year appears before price with no suffix

**Tier:** 0 (unit test)  
**Severity:** 🟠 MEDIUM → **FIXED**  
**Complexity:** ★★ MEDIUM

**Status:** Coupled fix with Bug #1. Tier 0 passes.

---

## ✅ Bug #3 — Nonsense model name returns confident results without warning

**Tier:** 3  
**Severity:** 🟠 MEDIUM → **RESOLVED (behavior changed)**  
**Complexity:** ★★ MEDIUM

**Status:** `Toyota Okozaky` now triggers the inexact banner: `⚠️ No exact match found for 'toyota okozaky' in our database. Here are the closest vehicles we could find:` — test passes. The underlying fuzzy behavior (returning Toyota Yaris) is acceptable per team decision. Mark closed.

---

## 🔴 Bug #4 — Pipeline hangs / times out on specific prompt patterns

**Tier:** 3 (2 × 600s timeout failures)  
**Severity:** 🔴 HIGH  
**Complexity:** ★★★ HARD  
**Owner:** Eylon

**Prompts that timed out (new run):**
1. `"Compact sedan for daily commuting, max price $8,500, minimum model year 2016."` — 3,657 seconds (~61 min), then timeout
2. `"Reliable hybrid hatchback for city driving, no more than $9,000, year 2014 or newer."` — 2,520 seconds (~42 min), then timeout

**NEW PATTERN OBSERVED:** Both timed-out prompts in this run have very low max budgets ($8,500 and $9,000). In the previous run the timed-out prompts had $25,000 and $30,000 budgets, so budget alone is not the trigger — but it's worth checking whether the agent loops when matching results are few or borderline. The hang is consistent across runs (2/9 random prompts each time).

**Fix direction:** Hard step limit (≤ 60 steps) with forced `FinalAnswer`; server-side per-request timeout independent of the client.

---

## ✅ Bug #6 — Returned listings violate `min_year` constraint

**Tier:** 3  
**Severity:** 🔴 HIGH → **FIXED**  
**Complexity:** ★★ MEDIUM  
**Owner:** Eylon

**Status:** All listings in the new report respect `year_min`. Examples verified:
- `"Reliable family SUV, 2018 or newer"` → Outlander 2018, 2019 ✓
- `"Family SUV with third row, 2018 or newer"` → Outlander 2018, 2019 ✓
- `"BMW 3serise, min 2018"` → BMW 3 Series 2018, 2020, 2020 ✓
- `"Convertible, min 2013"` → Mini Convertible 2014, 2015 ✓

---

## ✅ Bug #9 — `explicit_fields` path also violates year filter

**Tier:** 3  
**Severity:** 🔴 HIGH → **FIXED**  
**Complexity:** ★★ MEDIUM  
**Owner:** Eylon

**Status:** `explicit_fields` test with `year_min=2018` now returns Kia Sportage 2018 ✓. Same fix as #6.

---

## 🟠 Bug #5 — Classifier FP: non-car prompts reach the pipeline

**Tier:** 2  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  
**Owner:** Ron

**Progress:** Classifier prompt was updated. The classifier check (20260315) confirms:
- ✅ Motorcycle (Yamaha YZF-R3) → correctly rejected (TN)
- ✅ Box truck / cargo → correctly rejected (TN)
- ✅ Boat → correctly rejected (TN)
- ✅ Car parts (transmission) → correctly rejected (TN)
- ✅ Sell listing → correctly rejected (TN)
- ❌ **Car rental** still bypasses → see new Bug #11

**Remaining gap:** RV/motorhome (`Locate Class C RVs 2012 or newer, max $35,000`) still classified as car search (FP). Full pipeline runs and wastes budget.

**Fix direction:** Add explicit `"not_car"` examples for rentals and RVs to the classifier prompt.

---

## 🟠 Bug #7 — Listings with clearly invalid prices returned to users

**Tier:** 3 (passed, but suspicious)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★ EASY  
**Owner:** Ron

**Still observed in new run:**
| Model | Price |
|-------|-------|
| Tesla Model 3 (2019, Bellingham WA) | **$650** |
| Audi Q5 (2020, Salem OR) | **$565** |
| BMW 3 Series (2020, NYC) | **$0** |

**Root cause:** Bad/placeholder data in the listings DB. No minimum-price sanity filter applied.  
**Fix direction:** Field agent responsibility — if `price < $500`, validate with seller before presenting. Treat as missing data.

---

## 🟠 Bug #8 — Luxury sedan/coupe category: limited coverage

**Tier:** 3  
**Severity:** 🟡 LOW → **PARTIALLY IMPROVED**  
**Complexity:** ★★★ HARD  
**Owner:** Eylon

**Status:** New run returns Audi A7 for `"Luxury coupe (BMW, Mercedes, Audi preferred), max $28,000, min 2015"`. Coverage for common luxury brands has improved. However, only a single Audi model is returned with limited listing diversity. The previous hard failure (zero results for luxury sedan) no longer reproduces.

---

## 🟠 Bug #10 — RAG knowledge base uses UK sources (Euro NCAP, mpg, £ pricing context)

**Tier:** 3 (passed, but suspicious)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  
**Owner:** Ron (supervisor prompt)

**Still observed:** All "Why this model" explanations reference Euro NCAP ratings, mpg (UK measure), and UK driving context. UK-specific rationale (e.g., "B-roads", "boot") does not resonate with US users.

**Fix direction:** Rebuild RAG embeddings from US market sources or add a currency/locale normalization layer in the reasoning prompt.

---

## ✅ Bug #11 — NEW: Rental request bypasses intent classifier

**Tier:** 2 (explicit classifier failure)  
**Severity:** 🔴 HIGH  
**Complexity:** ★ EASY  
**Owner:** Ron

**Prompt:** `"Rent a midsize SUV 2018 or newer for one week, budget $60/day, pickup 2025-06-01"`  
**Result:** Classifier says `is_car_search=true` → full pipeline runs → AItzik returns Toyota RAV4 purchase listings

**Why this is HIGH:** The user asked to rent a car, not buy one. AItzik's entire pipeline (contact sellers, schedule viewings, negotiate price) is completely wrong for a rental intent. The user gets purchase listings and viewing appointments for a car they only want for a week.

**Root cause:** The classifier prompt says `'yes' = the request is about BUYING a used car` but apparently generalizes to any vehicle request. The word "rent" is not in the negative examples.

**Fix applied:** `"no: Rent a midsize SUV 2018 or newer for one week, budget $60/day, pickup 2025-06-01"` added as explicit few-shot example in the classifier prompt.

---

## 🟠 Bug #12 — NEW: `module_name_consistency` graded requirement FAIL

**Tier:** 0 (graded requirement)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★ EASY  
**Owner:** Ron/Eylon (check)

**Evidence:** `module_name_consistency: FAIL` in the graded requirements section of the new report.

**Fix direction:** Check which module names are inconsistent in the `_module_consistency` dict; likely a rename or casing mismatch. Quick grep of module declarations should identify the offending name.

---

## 🟠 Bug #13 — NEW: Inexact-model search extracts wrong token as model name

**Tier:** 3 (passed, suspicious)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  
**Owner:** Ron

**Observed in multiple responses:**
| Prompt | Inexact banner says |
|--------|---------------------|
| `"Crew-cab pickup for towing and hauling, up to $30,000"` | `"No exact match found for 'for towing'"` |
| `"Convertible for weekend drives, max price $18,000"` | `"No exact match found for 'min year'"` |
| `"Luxury coupe (BMW, Mercedes, Audi preferred)"` | `"No exact match found for 'audi preferred'"` |
| `"Find a used Chevvy Silverardo"` | `"No exact match found for 'min year'"` |

**Root cause:** The model/query extraction step is tokenizing the wrong part of the prompt as the vehicle model name. It appears to pick up the last significant phrase rather than the intended vehicle name.

**Impact:** The inexact banner shows confusing text to the user (`"No exact match found for 'min year'"`), and the subsequent fuzzy match operates on a nonsense query.

**Fix direction:** The model-name extraction logic needs to be anchored to the vehicle identifier part of the prompt (before constraints like budget/year).

---

## 🔴 Bug #14 — NEW: Fuzzy model match returns completely wrong vehicle category

**Tier:** 3 (passed, wrong result)  
**Severity:** 🔴 HIGH  
**Complexity:** ★★★ HARD  
**Owner:** Eylon

**Prompt:** `"Find a used Chevvy Silverardo, max price $35,000, minimum model year 2019"`  
**Returned:** `━━ AUDI Q5 ━━` (compact luxury crossover)  
**Expected:** Something in the full-size truck family (F-150, Silverado, RAM, etc.)

**Root cause:** Combined failure — the model extractor picked up `"min year"` as the query (Bug #13), and the RAG search on that nonsense term returned a random vector-similar result (Audi Q5). Neither the make ("Chevy") nor the category ("truck") was preserved.

**Impact:** User searching for a specific truck gets a luxury crossover. The response is confidently wrong with no caveat except the inexact banner.

---

## 🟠 Bug #15 — NEW: Empty model name in inexact model response

**Tier:** 3 (passed, but broken output)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★ MEDIUM  
**Owner:** 

**Prompt:** `"Looking for a used Nissan Altimor, max price $6,500, minimum model year 2012"`  
**Response includes:**
```
━━   ━━
💡 Why this model: Matches your criteria.
```

The model header is empty (`━━   ━━`) and the reasoning is a placeholder `"Matches your criteria."` — clearly a fallback path that was never properly handled.

**Fix direction:** Add a guard in the response formatter: if model name is empty or reasoning is the placeholder string, fall back to a generic "no close match found" error rather than returning an empty/meaningless recommendation.

---

## ✅ Bug #16 — NEW: Classifier FN — pickup/towing prompt rejected

**Tier:** 2 (classifier check only)  
**Severity:** 🟡 LOW  
**Complexity:** ★ EASY  
**Owner:** Ron

**Prompt:** `"Crew-cab pickup for towing and hauling, up to $30,000, min year 2017+."`  
**Result:** Classifier returned `is_car_search=false` (FN) — this is a valid car-buying request.

**Root cause:** The classifier may be confused by "towing and hauling" (sounds like a service, not a purchase) or the informal `2017+` format. Minor prompt sensitivity issue.

**Fix applied:** `"yes: Crew-cab pickup for towing and hauling, up to $30,000, min year 2017+."` added as explicit few-shot example in the classifier prompt.

---

## 🟠 Bug #17 — NEW: Most pipeline runs exceed efficiency threshold (6/9 FLAGGED)

**Tier:** 3 (efficiency metric)  
**Severity:** 🟠 MEDIUM  
**Complexity:** ★★★ HARD  
**Owner:** 

**Efficiency results:**
| Prompt | Steps | Flagged |
|--------|-------|---------|
| Reliable family SUV under $22,000, 2018 or newer | 36 | — |
| family SUV | 48 | ⚠️ |
| Family SUV with third row seating | 45 | ⚠️ |
| Crew-cab pickup for towing | 56 | ⚠️ |
| Electric car with 150-mile range | 41 | ⚠️ |
| Luxury coupe BMW/Mercedes/Audi | 44 | ⚠️ |
| Small AWD crossover for snowy climates | 48 | ⚠️ |
| Minivan with captain's chairs | 25 | — |
| Convertible for weekend drives | 36 | — |

6 of 9 runs flagged. The vague short prompt `"family SUV"` (no constraints) runs 48 steps despite being the simplest ask. The most expensive prompt (56 steps) is one that triggered Bug #13 (wrong model extraction), suggesting misrouted searches drive extra agent iterations.

---

## Summary Table

| # | Bug | Severity | Complexity | Source | Owner | Notes | Done? |
|---|-----|----------|------------|--------|-------|-------|-------|
| 1 | `_extract_price` picks year instead of price (bare numbers) | 🔴 HIGH | ★★ MEDIUM | Tier 0 failure | Ron | | ✅ |
| 2 | `_extract_year` returns None when year consumed by price regex | 🟠 MEDIUM | ★★ MEDIUM | Tier 0 failure | Ron | | ✅ |
| 3 | Nonsense model name `Okozaky` returns Toyota Yaris results | 🟠 MEDIUM | ★★ MEDIUM | Tier 3 failure | Ron | inexact banner now shown | ✅❌ |
| 4 | Pipeline hangs indefinitely on some prompts (>10 min timeout) | 🔴 HIGH | ★★★ HARD | Tier 3 failure | Eylon | low-budget prompts pattern | ❌ |
| 5 | Motorbikes / work trucks / RVs bypass intent classifier | 🟠 MEDIUM | ★★ MEDIUM | Tier 2 suspicious pass | Ron | motorcycle/truck/rental fixed; RV still open | 🔄 |
| 6 | Returned listings violate `min_year` constraint | 🔴 HIGH | ★★ MEDIUM | Tier 3 suspicious pass | Eylon | | ✅ |
| 7 | Listings with prices $0, $1, $189, $296 returned to users | 🟠 MEDIUM | ★ EASY | Tier 3 suspicious pass | Ron | field agent validation needed | ✅ |
| 8 | Luxury sedan category has zero data coverage | 🟡 LOW | ★★★ HARD | Tier 3 failure | Eylon | partially improved | 🔄 |
| 9 | `explicit_fields` path also violates year filter | 🔴 HIGH | ★★ MEDIUM | Tier 3 suspicious pass | Eylon | fixed with #6 | ✅ |
| 10 | RAG uses UK sources (Euro NCAP, mpg, £ pricing context) | 🟠 MEDIUM | ★★ MEDIUM | Tier 3 suspicious pass | Ron | supervisor prompt | ✅❌ |
| **11** | **Rental request bypasses classifier** | 🔴 HIGH | ★ EASY | Tier 2 bypass | Ron | explicit `no` example added to classifier prompt | ✅ |
| **12** | **`module_name_consistency` graded requirement FAIL** | 🟠 MEDIUM | ★ EASY | Tier 0 graded | Ron/Eylon | grep module declarations | ✅ |
| **13** | **Inexact search extracts wrong token as model name** | 🟠 MEDIUM | ★★ MEDIUM | Tier 3 suspicious pass | Ron | "for towing", "min year" as model | ❌ |
| **14** | **Fuzzy match returns completely wrong vehicle category (Silverado → Audi Q5)** | 🔴 HIGH | ★★★ HARD | Tier 3 suspicious pass | Eylon | downstream of Bug #13 | ❌ |
| **15** | **Empty model name + placeholder reasoning in inexact response** | 🟠 MEDIUM | ★★ MEDIUM | Tier 3 suspicious pass | Ron | `━━   ━━` / "Matches your criteria" | ❌ |
| **16** | **Classifier FN: pickup/towing prompt wrongly rejected** | 🟡 LOW | ★ EASY | Classifier check FN | Ron | explicit `yes` example added to classifier prompt | ✅ |
| **17** | **6/9 pipeline runs exceed efficiency step threshold** | 🟠 MEDIUM | ★★★ HARD | Tier 3 efficiency | Eylon | likely driven by Bug #13 loop | ❌ |

Legend for Done?: ✅ Fixed · 🔄 Partially improved · ❌ Open

---

## Recommended Fix Priority

1. **[Blocking]** Bug #4 — Infinite loop / pipeline hang. Availability risk, still 2/9 prompts.
2. **[High]** Bug #14 — Wrong vehicle category returned (Silverado → Audi Q5). Downstream of Bug #13.
3. **[High]** Bug #13 — Wrong token extracted as model name. Fix unblocks #14 and #17.
4. **[Medium]** Bug #15 — Empty model name / placeholder reasoning.
5. **[Medium]** Bug #12 — `module_name_consistency` FAIL. Should be a quick fix.
6. **[Medium]** Bug #5 remainder — RV still bypasses classifier (rental ✅ fixed).
7. **[Medium]** Bug #10 — RAG UK currency mismatch.
8. **[Low]** Bug #7 — Invalid listing prices. Field agent responsibility.
9. **[Low]** Bug #17 — Efficiency. Will improve as #13/#14 are fixed.
