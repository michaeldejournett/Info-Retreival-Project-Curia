# Curia

RaikesHacks 2026 — Team "Is There Input Length Validation?" (Michael, Will, Rishi)

An event discovery app with a **Looking For Group** feature — find UNL events and organize groups to attend together.

## Production

| Service | URL |
|---------|-----|
| **App (frontend)** | https://frontend-production-6a4e.up.railway.app |
| **API (backend)** | https://backend-production-21c6f.up.railway.app |
| **Scraper (FastAPI)** | https://api-production-a090.up.railway.app |

---

## Quick Start (local dev, 2 commands)

> **Prerequisites:** Node.js 18+ ([download](https://nodejs.org))

```bash
npm install    # installs root, backend, and frontend deps automatically
npm run dev    # starts both backend (port 3001) & frontend (port 5173) concurrently
```

Open **http://localhost:5173** — the app loads real UNL events from the bundled `scraped/events.json`. Event browsing, search, and group creation work out of the box.

### Enable Google Sign-In (optional)

Google Sign-In is required for groups and notifications. To enable it locally:

1. Copy the env template into place:
   ```bash
   cp .env.example backend/.env
   ```
2. Edit `backend/.env` and fill in your credentials:
   ```
   GOOGLE_CLIENT_ID=<your-client-id>
   GOOGLE_CLIENT_SECRET=<your-client-secret>
   SESSION_SECRET=<any-random-string>
   FRONTEND_URL=http://localhost:5173
   GOOGLE_CALLBACK_URL=http://localhost:3001/api/auth/google/callback
   ```
3. In [Google Cloud Console](https://console.cloud.google.com/), add `http://localhost:3001/api/auth/google/callback` to **Authorized redirect URIs**.

> **Never commit `backend/.env`** — it is gitignored.

---

## Running Everything via Docker

Docker adds live event scraping and Gemini-powered natural-language search.

```bash
docker compose up --build
```

| Service | Port | Description |
|---------|------|-------------|
| `backend` | `3001` | Express + SQLite (events, groups, auth API) |
| `api` | `8080` | FastAPI scraper + Gemini-powered search |

---

## Testing & Evaluation

### Unit tests

Two pytest suites live in `tests/`:

| File | What it covers |
|------|----------------|
| `tests/test_search.py` | Core search logic — music, sports, food, and case-insensitivity queries. Runs against `scraped/events.json` when present; falls back to in-file sample events otherwise. |
| `tests/test_search_accuracy.py` | Keyword expansion accuracy — validates that each category defined in `backend/db.js` returns more results with the expanded keyword set than with a single bare term. |

```bash
pip install -r api/requirements.txt pytest
python -m pytest tests/ -v
```

### Benchmark suite

The benchmark suite (`testing/benchmark/`) measures retrieval quality, latency, and robustness across model providers. It supports three backends (Gemini, Ollama, HuggingFace hosted/local) and five datasets.

**Prerequisites**

```bash
pip install -r api/requirements.txt
```

For HuggingFace hosted models, set `HF_TOKEN` in your environment. For local inference, also install `torch` and `transformers`.

**Datasets**

| Dataset | Cases | Focus |
|---------|-------|-------|
| `queries_smoke.json` | 5 | Sanity check — fast pass over the full pipeline |
| `queries_branch_compare_seed.json` | 8 | Seed cases for A/B branch comparisons |
| `queries_rigorous_retrieval.json` | 96 | Keyword and semantic retrieval |
| `queries_rigorous_temporal.json` | 72 | Date/time expression handling |
| `queries_rigorous_robustness.json` | 72 | Typos, partial matches, edge inputs |

**Running benchmarks via npm**

```bash
npm run benchmark:smoke          # quick sanity pass (default model set, smoke dataset)
npm run benchmark:correctness    # correctness gate on smoke dataset
npm run benchmark:latency        # latency-stability gate on smoke dataset
npm run benchmark:full           # full rigorous pass (all three datasets)
npm run benchmark:first-pass     # HuggingFace hosted models, smoke dataset
npm run benchmark:hf:router-chat # HuggingFace chat-router models
npm run benchmark:local-lite     # local CPU inference, smoke dataset
npm run benchmark:run-all        # all model sets, smoke dataset
```

**Running benchmarks directly**

```bash
# Smoke pass with a specific model
python -m testing.benchmark.run_benchmark --profile smoke --models gemini:gemma-3-27b-it

# Full rigorous pass with a custom dataset
python -m testing.benchmark.run_benchmark \
  --profile full \
  --dataset testing/benchmark/datasets/queries_rigorous_temporal.json \
  --models gemini:gemma-3-27b-it

# Local HuggingFace model
python -m testing.benchmark.run_benchmark \
  --profile smoke \
  --model-set hf-local-lite \
  --huggingface-backend local
```

**Output**

Each run creates a timestamped directory under `testing/benchmark/reports/` containing:

```
reports/<timestamp>/
├── summary.json          # aggregate scores per model
├── summary.md            # human-readable summary
├── per_query.csv         # per-query scores and latencies
├── gate_results.json     # pass/fail for each gate profile
├── gate_results.md
└── visuals/              # PNG charts (synced to figures/)
    ├── fig1_jaccard_similarity.png
    ├── fig2_temporal_accuracy.png
    ├── fig3_latency_cdf.png
    ├── fig4_quality_vs_speed.png
    └── fig5_per_query_heatmap.png
```

The five canonical figures are also copied to `figures/` at the project root after each run.

### Generate search metrics and charts

```bash
pip install matplotlib numpy
python scripts/generate_search_metrics.py   # writes metrics/ JSON + CSV
python scripts/plot_search_metrics.py       # writes metrics/ PNG charts
```

`generate_search_metrics.py` must run before the plot script. All output lands in `metrics/`. See [docs/graphs.md](docs/graphs.md) for a description of every chart.

---

## Tech Stack

- **Frontend:** React + Vite
- **Backend:** Node.js + Express + SQLite (events, groups, auth)
- **Scraper:** Python + FastAPI + Gemini API (UNL event scraping + LLM search — Docker only)
- **Local LLM pipeline (optional):** Two fine-tuned HuggingFace models running on CPU — `satyaalmasian/temporal_tagger_roberta2roberta` (~330M) for TIMEX3 temporal extraction and `google/flan-t5-base` (~250M) for semantic keyword expansion. Used as a local alternative to Gemini when `USE_LOCAL_MODELS=true`.

---

## Features

- **Event discovery** — Browse and filter UNL events by category, date, and location; results are paginated with a configurable page size
- **AI search** — Natural-language search via Gemini API; press Enter to submit. AI automatically extracts date/time ranges from queries and reflects them in the sidebar date picker
- **Keyword search** — Toggle off AI for instant debounced keyword search; date/time extraction is skipped and results are pure keyword matches
- **Search debug panel** — Collapsible info panel below the result count showing LLM used, expanded terms, detected date/time filters, and FastAPI connection status
- **Calendar export** — Download any event as an `.ics` file to add it directly to your calendar app
- **Looking For Group** — Create or join groups for any event, with capacity limits, meetup details, and vibe tags
- **Group messaging** — Real-time chat within groups (auto-refreshes every 3s), visible only to members
- **My Groups** — Quick-access menu in the navbar showing all groups you belong to
- **Notifications** — Bell icon tracks when someone joins/leaves your groups or sends a message
- **Google Sign-In** — OAuth 2.0 authentication via Google
- **Share links** — Copy a direct link to any group; recipients land on the event with the group visible

---

## Architecture

1. **Express API** (`backend/`) — Serves events, LFG groups, messages, and auth. Uses SQLite, zero config. Periodically pulls new events from the FastAPI scraper.
2. **FastAPI Scraper** (`api/`) — Scrapes real UNL events from events.unl.edu + Campus Labs Engage, with optional LLM-powered keyword expansion. Two backends supported:
   - **Gemini API** (default) — single call extracts keywords + date/time in one pass. Requires `GEMINI_API_KEY`.
   - **Local two-stage pipeline** (`USE_LOCAL_MODELS=true`) — runs fully offline on CPU:
     - *Stage 1* — [`api/temporal.py`](api/temporal.py) calls `satyaalmasian/temporal_tagger_roberta2roberta` to emit TIMEX3 tags, then [`api/timex3.py`](api/timex3.py) resolves each `value` attribute (`P2W`, `2026-W17-WE`, `TMO`, `FUTURE_REF`, etc.) into concrete ISO date/time ranges relative to today.
     - *Stage 2* — [`api/expansion.py`](api/expansion.py) calls `google/flan-t5-base` to expand keywords semantically.
   - Both paths fall through to `dateparser` + raw keyword matching if the LLMs are unavailable. Runs in Docker.
3. **Keyword Generalization** (`backend/keywords.js`) — At index time, generalizes event text into broader tags (e.g. "pizza" → food, "biology" → science) so searches find relevant events even when exact words don't match.
4. **Search scoring** (`backend/routes/events.js`) — Events are scored by field weight (name ×4, venue/category ×2, description/tags ×1) and by term position (terms appearing earlier in the query rank higher).
5. **Date filtering** — The sidebar calendar is the single source of truth. In AI mode, detected date ranges auto-populate the sidebar and can be further adjusted by the user. In keyword mode, date filters are not extracted from the query.
6. **ICS export** (`frontend/src/utils/icsGenerator.js`) — Generates RFC 5545-compliant `.ics` files client-side for any event.

---

## API Endpoints

### Events
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events` | List all events (includes group counts) |
| GET | `/api/events/:id` | Get a single event |
| GET | `/api/events/search?q=&no_llm=` | Keyword/AI search — returns matches ranked by relevance. Pass `no_llm=true` to skip LLM and use pure keyword matching |

### Groups
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/groups/mine` | List all groups the current user belongs to |
| GET | `/api/groups?eventId=` | List groups for an event |
| GET | `/api/groups/:id` | Get a single group with members |
| POST | `/api/groups` | Create a group |
| POST | `/api/groups/:id/join` | Join a group |
| POST | `/api/groups/:id/leave` | Leave a group |
| DELETE | `/api/groups/:id` | Delete a group (creator only) |
| GET | `/api/groups/:id/messages` | Get group messages |
| POST | `/api/groups/:id/messages` | Post a message |

### Notifications
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/notifications` | Get notifications for the current user (last 50 + unread count) |
| POST | `/api/notifications/read` | Mark all as read, or a single one with `{ id }` |

### Auth
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/auth/google` | Start Google OAuth flow |
| GET | `/api/auth/google/callback` | OAuth callback (handled automatically) |
| GET | `/api/auth/me` | Get current user (401 if not signed in) |
| POST | `/api/auth/logout` | Sign out |

---

## Environment Variables

All env vars go in `backend/.env` (created by `cp .env.example backend/.env`). The app works without them — Google Sign-In is simply disabled.

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLIENT_ID` | For auth | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | For auth | Google OAuth client secret |
| `GOOGLE_CALLBACK_URL` | Production | Full callback URL (e.g. `https://your-backend/api/auth/google/callback`) |
| `SESSION_SECRET` | For auth | Any random string for session signing |
| `FRONTEND_URL` | Production | Frontend origin for CORS (e.g. `https://your-frontend.app`) |
| `FASTAPI_URL` | Docker | URL of the FastAPI scraper service |
| `EVENTS_API_URL` | Docker | URL of the `/events` endpoint for periodic refresh |
| `REFRESH_INTERVAL_MS` | Optional | Event refresh interval in ms (default: 3600000 = 1 hour) |
| `NODE_ENV` | Production | Set to `production` to enable secure cookies and static serving |
| `GEMINI_API_KEY` | For AI search | Google Gemini API key — enables natural-language keyword expansion and date/time extraction. Falls back to raw keyword search if not set. Also accepted as `GOOGLE_API_KEY` |
| `GEMINI_MODEL` | Optional | Gemini model to use for search (default: `gemma-3-27b-it`) |
| `USE_LOCAL_MODELS` | Optional | `true` to route searches through the local two-stage LLM pipeline instead of Gemini (default: `false`) |
| `TEMPORAL_MODEL_ID` | Optional | HuggingFace ID for Stage 1 (default: `satyaalmasian/temporal_tagger_roberta2roberta`) |
| `EXPANSION_MODEL_ID` | Optional | HuggingFace ID for Stage 2 (default: `google/flan-t5-base`) |
| `LOCAL_MODEL_FALLBACK_TO_GEMINI` | Optional | `true` to retry with Gemini when local stack fails (default: `true`) |

---

## Security — What Not to Commit

The following are gitignored and must **never** be committed:

| File / Pattern | Why |
|----------------|-----|
| `backend/.env`, `.env`, `.env.local` | Contains OAuth secrets and session keys |
| `*.db`, `*.db-shm`, `*.db-wal` | SQLite database files may contain user data (names, emails) |
| `node_modules/`, `.venv/` | Dependencies — install locally, not stored in git |
| `dist/` | Build output — generated at deploy time |
| `backend/uploads/` | User-uploaded files |

If you accidentally commit a secret, rotate it immediately (generate a new Google OAuth secret, new session key, etc.).

---

## Deploying to Railway

Use the deploy script — it handles all three services:

```bash
./deploy.sh
```

Set Google OAuth credentials on the backend service (one-time):

```bash
railway variable --service backend set \
  GOOGLE_CLIENT_ID=<your-client-id> \
  GOOGLE_CLIENT_SECRET=<your-client-secret> \
  GOOGLE_CALLBACK_URL=https://<your-backend-domain>/api/auth/google/callback \
  SESSION_SECRET=<random-string> \
  FRONTEND_URL=https://<your-frontend-domain>
```

Set the Gemini API key on the scraper (`api`) service to enable AI search:

```bash
railway variable --service api set \
  GEMINI_API_KEY=<your-gemini-api-key>
```

Also add `https://<your-backend-domain>/api/auth/google/callback` to **Authorized redirect URIs** in Google Cloud Console.
