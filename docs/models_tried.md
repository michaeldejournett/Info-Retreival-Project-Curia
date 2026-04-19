# Models Tried — Curia Local Pipeline

Comparison context: UNL event search. Local pipeline does two stages — temporal extraction (date/time) and keyword expansion — then falls back to Gemini if local fails.

---

## Remote Baseline

### `gemma-3-27b-it` via Gemini API
- **Stage:** Both (temporal + expansion in one call)
- **Access:** `google-genai` SDK, `GEMINI_API_KEY`
- **Latency:** ~7.8s mean across 8 test queries
- **Keyword Jaccard vs itself:** 1.0 (baseline)
- **Temporal accuracy:** Correct on all 8 test queries
- **Notes:** Default `GEMINI_MODEL` env var. Reliable but slow and requires API key + network.

---

## Local Models Tried

### 1. `temporal_tagger_roberta2roberta` — **ABANDONED**
- **Stage:** Temporal extraction only
- **Size:** ~250M params encoder-decoder
- **Task it was trained on:** TIMEX3 annotation of news text
- **Failure mode:** Outputs bare `<TIMEX3 value="2026">` tags with no day/month anchoring; resolver inflated every query to full-year range 2026-01-01 → 2026-12-31. One query hallucinated `1998-W44-WE`.
- **Root cause:** Trained to *tag* dates inside existing news documents, not to *resolve* relative expressions like "two weeks from now" against a current date context prefix.
- **Result:** 0/8 date range matches vs Gemini.

### 2. `google/flan-t5-base` — **ABANDONED**
- **Stage:** Keyword expansion only
- **Size:** ~250M params encoder-decoder
- **Task it was trained on:** Multi-task instruction following (FLAN fine-tune on T5)
- **Failure mode:** Zero-shot prompting produced alphabetic enumeration ("a, b, c, d…") or topic-irrelevant literals ("april25").
- **Root cause:** At 250M params, zero-shot structured JSON output is unreliable. The model has no chat template; applying few-shot with the T5 seq2seq format would have required custom prompt engineering.
- **Result:** Jaccard 0.00 on all 8 queries.

### 3. `Qwen/Qwen2.5-0.5B-Instruct` — **Current (baseline)**
- **Stage:** Both (temporal + expansion, separate prompts)
- **Size:** ~500M params decoder-only, ~954MB weights
- **Format:** Chat-templated (`apply_chat_template`), few-shot exemplars, greedy decoding
- **Temporal prompt:** 4 exemplars with CoT "Reasoning: …" prefix + dateparser fallback
- **Expansion prompt:** 3 exemplars, 20-25 keywords each, hybrid with MiniLM retrieval
- **Latency:** ~3.2s mean (faster than Gemini)
- **Keyword Jaccard vs Gemini:** 0.00 (initial run before few-shot); improved but still low
- **Temporal accuracy:** Partially correct — distinct failure modes:
  - **Week boundary confusion:** "this week" anchored to current weekday, not Mon–Sun
  - **Day-of-week arithmetic:** "thursday" (when today=Friday) → returned today's date instead of prior/next Thursday
  - **Partial time range:** "sports games tonight" → `time_to: "21:00"` only, `time_from: null`
  - **Hallucinated logic:** "pizza on April 25" → invented "the day before April 25" reasoning
- **Root cause:** 0.5B parameter ceiling insufficient for multi-step calendar arithmetic even with CoT.
- **Status:** Running in production as default `LOCAL_MODEL_ID`.

### 4. `sentence-transformers/all-MiniLM-L6-v2` — **Active (retrieval hybrid)**
- **Stage:** Keyword expansion augmentation only (not standalone)
- **Size:** ~22M params, ~90MB
- **Role:** Embeds query seeds + pre-embedded 384-term corpus from `backend/keywords.js`; cosine similarity retrieval adds semantically related terms to LLM output
- **Latency:** <50ms (embedding lookup, cached corpus)
- **Notes:** Loaded separately via `api/retrieval.py`. Always on when retrieval model loads, regardless of which LLM is used.

### 5. `Qwen/Qwen2.5-1.5B-Instruct` — **Tested (2026-04-17)**
- **Stage:** Both (same prompts as 0.5B, no code changes)
- **Size:** ~1.5B params, ~3GB weights
- **Latency:** ~129s mean per query (CPU, 4-core) — 14× slower than Gemini, unusable in practice
- **Keyword Jaccard vs Gemini:** 0.18 mean (up from ~0.00 on 0.5B)
- **Temporal accuracy:** 3/7 date matches, 2/2 time matches
  - ✅ "this weekend concerts" → Apr 18–19 correct
  - ✅ "art exhibits next month" → May 1–31 correct
  - ✅ "sports games tonight" → Apr 17, 17:00–21:00 correct
  - ✅ "morning events this week" → time 06:00–12:00 correct (date still off: Apr 17 only)
  - ❌ "free food thursday" → returned today (Apr 17) instead of next Thu (Apr 23)
  - ❌ "two weeks from now" date queries → disagreed with Gemini's window interpretation
  - ❌ "pizza on April 25" → no date extracted at all
- **vs 0.5B:** Reasoning quality clearly better (time ranges now perfect, weekend/month correct). Remaining failures are edge cases, not systematic breakdowns.
- **Verdict:** Solves the reasoning ceiling but CPU latency makes it impractical. Would need GPU or quantized (GGUF/int4) to be viable.
- **Status:** Tested, not deployed.

---

## Summary Table

| Model | Stage | Params | Temporal Acc | KW Jaccard | Latency | Status |
|---|---|---|---|---|---|---|
| `gemma-3-27b-it` (Gemini API) | Both | 27B (remote) | ✅ 8/8 | baseline | ~7.8s | Active (remote) |
| `temporal_tagger_roberta2roberta` | Temporal | ~250M | ❌ 0/8 | — | — | Abandoned |
| `flan-t5-base` | Expansion | ~250M | — | ❌ 0.00 | — | Abandoned |
| `Qwen2.5-0.5B-Instruct` | Both | 500M | ⚠️ partial | low | ~3.2s | Active (local default) |
| `all-MiniLM-L6-v2` | Expansion (hybrid) | 22M | — | augments LLM | <50ms | Active |
| `Qwen2.5-1.5B-Instruct` | Both | 1.5B | ⚠️ 3/7 dates, 2/2 times | 0.18 | ~129s | Tested, not deployed |

---

## Known Failure Modes (0.5B Temporal)

| Failure | Example Query | Expected | Actual |
|---|---|---|---|
| Week boundary | "morning events this week" | Mon–Fri | Fri–Thu (anchored to current day) |
| Day-of-week arithmetic | "free food thursday" | next/prev Thu | today's date |
| Partial time range | "sports games tonight" | 17:00–21:00 | null→21:00 |
| Hallucinated logic | "pizza on April 25" | Apr 25 | Apr 24 (invented "day before" logic) |

---

## Env Vars

| Var | Default | Purpose |
|---|---|---|
| `LOCAL_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Swap local model without code change |
| `USE_LOCAL_MODELS` | `false` | Enable local pipeline |
| `GEMINI_MODEL` | `gemma-3-27b-it` | Remote baseline model |
| `GEMINI_API_KEY` | — | Required for remote baseline |
