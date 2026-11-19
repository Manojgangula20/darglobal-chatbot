# DarGlobal & Wasalt Assistant

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)

An AI chatbot that answers questions about **DarGlobal** and **Wasalt** using only what their public websites say. Every answer links to the pages it came from, and when the data does not cover a question, the bot says so instead of guessing.

**Live demo:** https://darglobal-chatbot-fhf9.onrender.com

---

## Highlights

| | |
|---|---|
| **Grounded answers** | Answers come only from scraped website content. If it is not there, the bot replies "I don't have that information". |
| **Verifiable sources** | The model tags which pages it used, so the links under each answer are the pages the answer actually came from. |
| **Hybrid search** | Local semantic embeddings plus BM25 keyword search, merged with reciprocal rank fusion. Finds pages by meaning and by exact terms. |
| **Always fresh** | Hourly refresh with change detection. Only changed pages are re-embedded. |
| **Safe updates** | A new index is built in full, then swapped in. A failed or suspiciously small scrape never replaces good data. |
| **Free to run** | Local embeddings (no API key) and a free OpenRouter model with automatic fallbacks. |

---

## Example questions

- What does DarGlobal do?
- What is Wasalt?
- What projects does DarGlobal have in Oman?
- What perks come with owning an AIDA residence?

Questions the public sites do not answer (for example exact registration fees or projected yields) get an honest "I don't have that information" reply.

---

## How it works

```
Scrape (DarGlobal, Wasalt, Wasalt blog)
        |
        v
Clean text -> chunks -> local embeddings -> in-memory index (+ BM25 keywords)
        |
        v
User question -> hybrid search -> top-k chunks -> OpenRouter free model
        |
        v
Answer + source links
```

### Data sources

| Source | Method |
|---|---|
| DarGlobal (`darglobal.co.uk`) | English pages via sitemap and link crawl |
| Wasalt (`wasalt.sa`) | English pages via sitemap and link crawl; property listings, login and payment pages are skipped |
| Wasalt blog (`blog.wasalt.sa`) | WordPress REST API, latest posts |

The scraper respects `robots.txt`, waits between requests, and identifies itself with a clear User-Agent.

### Keeping data up to date

| What | How often |
|---|---|
| In-app refresh | Every `REFRESH_INTERVAL_MIN` minutes (default 60) |
| Safety net | `/health` starts a background refresh if the data is older than the interval |
| Manual refresh | `POST /refresh` with header `X-Refresh-Token` |
| Keep-alive (optional) | Cloudflare Workers cron in `cloudflare-cron/` pings `/health` so the free instance does not sleep |

Change detection uses conditional requests (ETag / Last-Modified) plus a SHA-256 hash of each page's text.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, Uvicorn, APScheduler |
| Scraping | requests, BeautifulSoup, lxml |
| Embeddings | `all-MiniLM-L6-v2` through FastEmbed (ONNX, runs locally) |
| Search | NumPy cosine similarity + BM25, fused with reciprocal rank fusion |
| LLM | Any free model on OpenRouter, with fallbacks |
| UI | Single static page, no framework |
| Packaging | Docker |
| Hosting | Render |

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Chat interface |
| `POST /chat` | Ask a question. Body: `{"message": "...", "history": []}` |
| `GET /health` | Liveness check (also triggers a refresh if data is stale) |
| `GET /status` | Last refresh time, chunks per source, last result |
| `POST /refresh` | Force a refresh. Requires header `X-Refresh-Token` |

---

## Quick start

### Run locally

**PowerShell (Windows)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m app.cli dry-run        # preview what will be scraped, saves nothing
python -m app.cli scrape         # full scrape -> seed/pages.json

$env:OPENROUTER_API_KEY = "sk-or-..."
python -m app.cli ask "What is Wasalt?"
uvicorn app.main:app --port 7860
```

**Bash (macOS / Linux)**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m app.cli dry-run
python -m app.cli scrape

export OPENROUTER_API_KEY=sk-or-...
python -m app.cli ask "What is Wasalt?"
uvicorn app.main:app --port 7860
```

Then open http://localhost:7860.

### Run with Docker

```bash
docker build -t darglobal-bot .
docker run -p 7860:7860 \
  -e OPENROUTER_API_KEY=sk-or-... \
  -e REFRESH_TOKEN=choose-a-secret \
  darglobal-bot
```

### Deploy on Render

1. Create a **Web Service** on [Render](https://render.com/) and connect this repository.
2. Select **Docker** as the runtime and `main` as the branch.
3. Add the environment variables below.
4. Deploy, then open the generated URL.
5. Optional: set Render's health check path to `/health`.

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `REFRESH_TOKEN` | Recommended | Protects `POST /refresh` |
| `MODEL_PRIMARY` | No | Primary OpenRouter model (use a `:free` model) |
| `MODEL_FALLBACKS` | No | Comma-separated fallback models |
| `REFRESH_INTERVAL_MIN` | No | Minutes between automatic refreshes (default 60) |
| `TOP_K` | No | Chunks passed to the model (default 7, 10 recommended) |

See `.env.example` for a template.

---

## Project structure

```
.
├── app/
│   ├── main.py          FastAPI app, endpoints, scheduler
│   ├── scraper.py       Crawling, robots.txt, change detection, blog API
│   ├── indexer.py       Chunking, embeddings cache, hybrid search
│   ├── rag.py           Prompt, OpenRouter calls with fallback, source selection
│   ├── refresh.py       Refresh job and safe index swap
│   ├── store.py         Snapshot storage
│   ├── cli.py           dry-run / scrape / build / ask
│   ├── config.py        Settings and site definitions
│   └── static/index.html
├── seed/pages.json      Committed data snapshot used at startup
├── cloudflare-cron/     Optional keep-alive worker
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Design decisions

- **Committed snapshot:** `seed/pages.json` lets the app answer immediately after any restart, then refresh itself.
- **Local embeddings:** no external embedding API, which keeps the project free and the memory footprint small (the target is about 512 MB RAM).
- **Hybrid search:** semantic search alone missed pages that use exact words such as "membership" or "concierge", so keyword search is merged in.
- **Attribution:** marketing claims (rankings, projections, benefits) are presented as the company's own statements.
- **Refresh safety:** if a scrape fails or returns far less content than before, the previous data stays live.

---

## Known limitations

- Free hosting storage can be ephemeral, so runtime caches may be rebuilt after a restart. The committed snapshot covers this.
- The data is a snapshot of public pages. It can lag behind the websites until the next refresh.
- Free OpenRouter models can be rate-limited or briefly unavailable. Fallback models are tried automatically.
- The index and refresh state are designed for a single application instance.
- The first request after a long idle period on a free instance can be slow.

---

## Assignment requirements

| Requirement | How it is met |
|---|---|
| Scrape public data from DarGlobal and Wasalt | `app/scraper.py` collects pages from both sites and the Wasalt blog |
| Build an AI chatbot using the collected data | Retrieval-augmented chatbot with sourced answers |
| Use a free OpenRouter model | Configurable `:free` models with fallbacks |
| Containerise with Docker | `Dockerfile` in this repository |
| Deploy, not only source code | Docker web service on Render |
| Provide a working URL | See **Live demo** at the top |