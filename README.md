# Curia

This is the RaikesHacks 2026 project for the team "Is There Input Length Validation?" consisting of Michael, Will, and Rishi.

An event discovery app with a **Looking For Group** feature — find UNL events and organize groups to attend together.

# Hosted at:

# https://frontend-production-6a4e.up.railway.app/


## Local Quick Start

> **Prerequisites:** Node.js 18+ ([download](https://nodejs.org))

```bash
npm install    # installs root, backend, and frontend deps automatically
npm run dev    # starts both backend & frontend concurrently
```

Open **http://localhost:5173** — the app loads real UNL events from the bundled `scraped/events.json` (no Docker or external services needed). Event browsing, search, and group creation all work out of the box.

**Google Sign-In** is needed for groups/notifications. To enable it, copy the env template and fill in credentials:

```bash
cp .env.example backend/.env
# edit backend/.env with your Google OAuth credentials
```

**Docker** adds live event scraping and LLM-powered natural-language search via FastAPI + Ollama. See [Running Everything via Docker](#running-everything-via-docker) for more details below.

```bash
# Mac / CPU-only / Windows
docker compose up --build

# Linux + NVIDIA GPU
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build

# Linux + AMD GPU (ROCm)
docker compose -f docker-compose.yml -f docker-compose.amd.yml up --build
```

## Tech Stack

- **Frontend:** React + Vite
- **Backend:** Node.js + Express + SQLite (events, groups, auth API)
- **Scraper:** Python + FastAPI + Ollama (UNL event scraping + LLM search — runs via Docker)

## Features

- **Event discovery** — Browse and filter UNL events by category, date, and location
- **AI search** — Natural-language search with keyword generalization (debounced, Enter to submit)
- **Looking For Group** — Create or join groups for any event, with capacity limits, meetup details, and vibe tags
- **Group messaging** — Real-time chat within groups (auto-refreshes every 3s), visible only to members
- **My Groups** — Quick-access menu in the navbar showing all groups you belong to
- **Notifications** — Bell icon tracks when someone joins/leaves your groups or sends a message
- **Google Sign-In** — OAuth 2.0 authentication via Google
- **Share links** — Copy a direct link to any group; recipients land on the event with the group visible

## Architecture

1. **Express API** (`backend/`) — Serves events, LFG groups, messages, and auth. Uses SQLite, zero config. Periodically pulls new events from the FastAPI scraper.
2. **FastAPI Scraper** (`api/`) — Scrapes real UNL events from events.unl.edu + Campus Labs Engage, with optional LLM-powered keyword expansion via Ollama. Runs in Docker.
3. **Keyword Generalization** (`backend/keywords.js`) — At index time, generalizes event text into broader tags (e.g. "pizza" → food, "biology" → science) so searches find relevant events even when exact words don't match.

## API Endpoints

### Events
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events` | List all events (includes group counts) |
| GET | `/api/events/:id` | Get a single event |
| GET | `/api/events/search?q=` | AI keyword search — returns all matches ranked by relevance |

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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLIENT_ID` | Yes (for auth) | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Yes (for auth) | Google OAuth client secret |
| `GOOGLE_CALLBACK_URL` | Production | Full callback URL (e.g. `https://your-domain/api/auth/google/callback`) |
| `SESSION_SECRET` | Yes | Random string for session signing |
| `FASTAPI_URL` | No | URL of the FastAPI scraper service |
| `EVENTS_API_URL` | No | URL of the `/events` endpoint for periodic refresh |
| `REFRESH_INTERVAL_MS` | No | How often to pull new events (default: 3600000 = 1 hour) |
| `NODE_ENV` | Production | Set to `production` to enable secure cookies and static serving |

## Running Everything via Docker

`docker compose up --build` starts all three services together:

| Service | Port | Description |
|---------|------|-------------|
| `backend` | `3001` | Express + SQLite (events, groups, auth API) |
| `api` | `8080` | FastAPI scraper + LLM search |
| `ollama` | — | Local LLM inference (internal only) |

```bash
# Mac / CPU-only / Windows
docker compose up --build

# Linux + NVIDIA GPU
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build

# Linux + AMD GPU (ROCm)
docker compose -f docker-compose.yml -f docker-compose.amd.yml up --build
```

## Deploying to Railway

Use the deploy script — it handles both services and wires up environment variables automatically:

```bash
./deploy.sh
```

Then set your Google OAuth credentials on the backend service (one-time):

```bash
railway variable --service backend set \
  GOOGLE_CLIENT_ID=<your-client-id> \
  GOOGLE_CLIENT_SECRET=<your-client-secret> \
  GOOGLE_CALLBACK_URL=https://<your-backend-domain>/api/auth/google/callback \
  SESSION_SECRET=<random-string>
```

Also add `https://<your-backend-domain>/api/auth/google/callback` to your **Authorized redirect URIs** in Google Cloud Console.
