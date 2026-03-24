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

## Tech Stack

- **Frontend:** React + Vite
- **Backend:** Node.js + Express + SQLite (events, groups, auth)
- **Scraper:** Python + FastAPI + Gemini API (UNL event scraping + LLM search — Docker only)

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
2. **FastAPI Scraper** (`api/`) — Scrapes real UNL events from events.unl.edu + Campus Labs Engage, with optional LLM-powered keyword expansion via the Gemini API. Requires a `GEMINI_API_KEY`; falls back to raw keyword matching if absent. Runs in Docker.
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
