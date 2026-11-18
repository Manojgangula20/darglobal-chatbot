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
| Keep-alive ping to `/health` (Cloudflare Workers cron) | every 15 minutes |
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

## Deploy (Hugging Face Spaces)
1. Create a Space, SDK **Docker**, blank template.
2. Space → Settings → Variables and secrets: add secrets `OPENROUTER_API_KEY` and `REFRESH_TOKEN`.
3. Push this repo to the Space's git remote.
4. Live URL: `https://<username>-<space-name>.hf.space`
5. Keep-alive: `cd cloudflare-cron`, set `APP_URL` in `wrangler.toml`, run `npx wrangler deploy`.

## Notes and limits
- Free hosting disk is ephemeral; the committed `seed/pages.json` snapshot makes the bot work immediately after any restart, then it refreshes itself.
- Free OpenRouter models can be rate-limited; fallbacks are tried automatically.
- Data is a snapshot of public pages; the answer sources are linked so it can be verified.
