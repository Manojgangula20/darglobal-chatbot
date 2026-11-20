from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config as C, rag, refresh as R
from .batch_runner import build_batch_runner
from .scheduler import BatchRefreshScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
STATIC = Path(__file__).parent / "static"

CHAT_LIMIT = 12
CHAT_WINDOW = 60
chat_requests = defaultdict(deque)

batch_scheduler: BatchRefreshScheduler | None = None


def _get_batch_targets(batch_index: int, batch_size: int) -> list[tuple[str, str]]:
    if hasattr(R, "get_batch_targets"):
        return R.get_batch_targets(batch_index, batch_size)
    return []


def _scrape_page(site: str, url: str) -> str:
    if hasattr(R, "scrape_single_page"):
        return R.scrape_single_page(site, url)
    return ""


def _upsert_chunks(chunks: list[dict]) -> int:
    if hasattr(R, "upsert_batch_chunks"):
        return R.upsert_batch_chunks(chunks)
    return 0


def _get_chunk_counts() -> tuple[int, dict[str, int]]:
    s = R.status()
    total = int(s.get("chunks", 0))
    by_site = dict(s.get("chunks_by_site", {}))
    return total, by_site


@asynccontextmanager
async def lifespan(app):
    global batch_scheduler
    R.load_initial()
    s = R.status()
    runner = build_batch_runner(
        get_batch_targets_fn=_get_batch_targets,
        scrape_page_fn=_scrape_page,
        upsert_chunks_fn=_upsert_chunks,
        get_chunk_counts_fn=_get_chunk_counts,
    )
    batch_scheduler = BatchRefreshScheduler(
        num_batches=C.NUM_BATCHES,
        batch_size=C.BATCH_SIZE,
        gap_min=C.BATCH_GAP_MIN,
        refresh_interval_min=C.BATCH_REFRESH_INTERVAL_MIN,
        run_batch_fn=runner,
        initial_chunks=int(s.get("chunks", 0)),
        initial_chunks_by_site=dict(s.get("chunks_by_site", {})),
    )
    batch_scheduler.start()
    yield
    if batch_scheduler is not None:
        batch_scheduler.stop()


app = FastAPI(title="DarGlobal & Wasalt Assistant", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[dict] = []


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    if batch_scheduler is not None:
        s = batch_scheduler.get_status()
        if not s["running"] and any(
            (time.time() - b["last_refreshed_ts"]) >= C.BATCH_REFRESH_INTERVAL_MIN * 60
            for b in s["batches"]
        ):
            batch_scheduler.trigger_batch(None)
    else:
        s = R.status()
    return {"status": "ok", "last_checked": s["last_checked"], "chunks": s["chunks"]}


@app.get("/status")
def status():
    if batch_scheduler is not None:
        return batch_scheduler.get_status()
    return R.status()


@app.post("/refresh")
def manual_refresh(batch_index: int | None = None, x_refresh_token: str = Header(default="")):
    if not C.REFRESH_TOKEN or x_refresh_token != C.REFRESH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")
    if batch_scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler not initialized")
    return batch_scheduler.trigger_batch(batch_index=batch_index)


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
    now = time.time()
    q = chat_requests[ip]
    while q and now - q[0] > CHAT_WINDOW:
        q.popleft()
    if len(q) >= CHAT_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    q.append(now)
    idx = R.INDEX
    if idx is None or not idx.chunks:
        return {"answer": "I'm still loading my data. Please try again in a minute.", "sources": []}
    return rag.answer(body.message, body.history[-6:], idx)