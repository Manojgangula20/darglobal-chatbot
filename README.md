---
title: DarGlobal Wasalt Assistant
emoji: 🏙️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DarGlobal & Wasalt Assistant

A RAG chatbot that answers questions about **DarGlobal** and **Wasalt** using only their public website content.
It refreshes its own data automatically and never serves stale answers for long.

## How it works
```
scraper → clean text → chunks → local embeddings → in-memory index
user question → top-k chunks → OpenRouter free model → answer + source links
```
- **Scraping** (`app/scraper.py`): DarGlobal and Wasalt English pages (sitemap + link crawl), plus the Wasalt blog via its WordPress REST API. Respects `robots.txt`, uses a polite delay and a clear User-Agent, skips property listings, login and payment pages.
- **Change detection**: conditional GET (ETag / Last-Modified) plus a SHA-256 hash of each page's text. Only changed pages are re-embedded (embedding cache keyed by chunk hash).
- **Safe refresh**: the new index is built fully, then swapped in atomically. If a site is down or a scrape returns far less than before, the previous data is kept.
- **Embeddings**: `all-MiniLM-L6-v2` via fastembed (ONNX, local, no API key). Cosine search with NumPy (the corpus is small).
- **LLM**: OpenRouter free model, configurable by env var, with automatic fallback models and retries.
- **Grounding**: strict prompt, answers only from retrieved context, otherwise says it doesn't have the information. Sources are shown under each answer.

## Freshness

| What | How often |
|---|---|
| In-app data refresh | every `REFRESH_INTERVAL_MIN` (default 60) minutes |
| Safety net | `/health` starts a background refresh if data is older than the interval |
| Manual | `POST /refresh` with header `X-Refresh-Token` |

## Endpoints
`GET /` chat UI · `POST /chat` · `GET /health` · `GET /status` · `POST /refresh`

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli dry-run          # check what will be scraped (8 pages per site, nothing saved)
python -m app.cli scrape           # full scrape -> seed/pages.json
export OPENROUTER_API_KEY=sk-or-...
python -m app.cli ask "What is Wasalt?"
uvicorn app.main:app --port 7860   # open http://localhost:7860
```

## Docker
```bash
docker build -t darglobal-bot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... -e REFRESH_TOKEN=secret darglobal-bot
```

## Deploy (Render)

The application is deployed as a Docker web service on Render.

1. Create a new Web Service on [Render](https://render.com/).
2. Connect this GitHub repository.
3. Select **Docker** as the runtime.
4. Set the deployment branch to `main`.
5. Add the following environment variables under Environment:

   - `OPENROUTER_API_KEY` — required, OpenRouter API key.
   - `REFRESH_TOKEN` — token used to protect manual refresh requests.
   - `MODEL_PRIMARY` — primary OpenRouter model (optional).
   - `MODEL_FALLBACKS` — comma-separated fallback models (optional).
   - `REFRESH_INTERVAL_MIN` — automatic refresh interval in minutes (default: 60).

6. Deploy the service. Render builds the Docker image and starts the application.
7. Open the generated Render URL.

### Health check

Render can monitor the application using:

`/health`

The application also exposes:

- `/` — chatbot interface.
- `/chat` — chatbot API.
- `/status` — index and refresh status.
- `/refresh` — manually trigger a refresh (requires `X-Refresh-Token`).

### Resource constraints and design decisions

The application targets a lightweight hosting environment (approximately 512 MB RAM and 0.1 CPU).

- **Local embeddings:** `all-MiniLM-L6-v2` runs through FastEmbed and ONNX. No external embedding API is required.
- **Committed snapshot:** `seed/pages.json` provides the initial website data so the application can initialize without requiring a successful live scrape during startup.
- **Incremental processing:** Unchanged pages reuse cached embeddings.
- **Atomic index replacement:** A new index is built before replacing the active index.
- **Refresh safety:** If scraping fails or returns substantially less content than expected, the previous index remains available.
- **LLM usage:** OpenRouter free models are subject to provider rate limits and availability. Fallback models can be configured through environment variables.

### Known limitations

- Render filesystem storage may be ephemeral, so runtime data and embedding caches can be lost after a restart.
- The committed snapshot can become outdated until the next successful refresh.
- Free OpenRouter models may return HTTP 429 or experience temporary outages.
- The in-memory index and refresh state are designed for a single application instance.
- Answers are limited to the scraped public website content, with source links provided for verification.

## Notes and limits
- Free hosting disk is ephemeral; the committed `seed/pages.json` snapshot makes the bot work immediately after any restart, then it refreshes itself.
- Free OpenRouter models can be rate-limited; fallbacks are tried automatically.
- Data is a snapshot of public pages; the answer sources are linked so it can be verified.
