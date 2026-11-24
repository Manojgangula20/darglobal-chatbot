import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import hmac
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
log = logging.getLogger("main")
STATIC = Path(__file__).parent / "static"

CHAT_LIMIT = 12
CHAT_WINDOW = 60
MAX_TRACKED_IPS = 2000
chat_requests: dict[str, deque[float]] = defaultdict(deque)

batch_scheduler: BatchRefreshScheduler | None = None
_scheduler_lock = asyncio.Lock()


def _get_batch_targets(batch_index: int, batch_size: int) -> list[tuple[str, str]]:
    return R.get_batch_targets(batch_index, batch_size) if hasattr(R, "get_batch_targets") else []


def _scrape_page(site: str, url: str) -> str:
    return R.scrape_single_page(site, url) if hasattr(R, "scrape_single_page") else ""


def _upsert_chunks(chunks: list[dict]) -> int:
    return R.upsert_batch_chunks(chunks) if hasattr(R, "upsert_batch_chunks") else 0


def _get_chunk_counts() -> tuple[int, dict[str, int]]:
    s = R.status()
    return int(s.get("chunks", 0)), dict(s.get("chunks_by_site", {}))


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip() or "?"
    return request.client.host if request.client and request.client.host else "?"


def _prune_rate_limiter(now: float) -> None:
    stale_ips = [ip for ip, q in chat_requests.items() if not q or now - q[0] > CHAT_WINDOW]
    for ip in stale_ips:
        q = chat_requests[ip]
        while q and now - q[0] > CHAT_WINDOW:
            q.popleft()
        if not q:
            del chat_requests[ip]
    if len(chat_requests) > MAX_TRACKED_IPS:
        oldest_ips = sorted(
            chat_requests.keys(),
            key=lambda k: chat_requests[k][-1] if chat_requests[k] else 0.0,
        )[: len(chat_requests) - MAX_TRACKED_IPS]
        for ip in oldest_ips:
            chat_requests.pop(ip, None)


async def _init_runtime() -> None:
    global batch_scheduler
    try:
        await asyncio.to_thread(R.load_initial)
        s = R.status()
        runner = build_batch_runner(
            get_batch_targets_fn=_get_batch_targets,
            scrape_page_fn=_scrape_page,
            upsert_chunks_fn=_upsert_chunks,
            get_chunk_counts_fn=_get_chunk_counts,
        )
        scheduler = BatchRefreshScheduler(
            num_batches=C.NUM_BATCHES,
            batch_size=C.BATCH_SIZE,
            gap_min=C.BATCH_GAP_MIN,
            refresh_interval_min=C.BATCH_REFRESH_INTERVAL_MIN,
            run_batch_fn=runner,
            initial_chunks=int(s.get("chunks", 0)),
            initial_chunks_by_site=dict(s.get("chunks_by_site", {})),
        )
        init_ts = time.time()
        batches = scheduler.state.get("batches", [])
        for b in batches:
            if b.get("last_refreshed_ts", 0.0) == 0.0:
                b["last_refreshed_ts"] = init_ts
        scheduler.start()
        async with _scheduler_lock:
            batch_scheduler = scheduler
        log.info("Background runtime initialized (chunks=%s)", s.get("chunks", 0))
    except asyncio.CancelledError:
        log.info("Background runtime initialization cancelled")
        raise
    except Exception:
        log.exception("Background runtime initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_scheduler
    task = asyncio.create_task(_init_runtime())
    app.state.init_task = task
    try:
        yield
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        async with _scheduler_lock:
            sched = batch_scheduler
            batch_scheduler = None
        if sched is not None:
            try:
                sched.stop()
            except Exception:
                log.exception("Failed stopping batch scheduler cleanly")


app = FastAPI(title="DarGlobal & Wasalt Assistant", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[dict] = []


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    sched = batch_scheduler
    s = sched.get_status() if sched is not None else R.status()
    return {"status": "ok", "last_checked": s.get("last_checked"), "chunks": s.get("chunks", 0)}


@app.get("/status")
def status():
    sched = batch_scheduler
    return sched.get_status() if sched is not None else R.status()


@app.post("/refresh")
def manual_refresh(batch_index: int | None = None, x_refresh_token: str = Header(default="")):
    if not C.REFRESH_TOKEN or not hmac.compare_digest(
        x_refresh_token.encode("utf-8"), C.REFRESH_TOKEN.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid token")
    sched = batch_scheduler
    if sched is None:
        raise HTTPException(status_code=503, detail="scheduler not initialized")
    return sched.trigger_batch(batch_index=batch_index)


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    ip = _client_ip(request)
    now = time.time()
    _prune_rate_limiter(now)
    q = chat_requests[ip]
    while q and now - q[0] > CHAT_WINDOW:
        q.popleft()
    if len(q) >= CHAT_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    q.append(now)

    idx = R.INDEX
    if idx is None or not idx.has_data:
        return {"answer": "I'm still loading my data. Please try again in a minute.", "sources": []}
    return rag.answer(body.message, body.history[-6:], idx)