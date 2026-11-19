import logging
from contextlib import asynccontextmanager
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config as C, rag, refresh as R

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
STATIC = Path(__file__).parent / "static"
sched = BackgroundScheduler(timezone="UTC")
from collections import defaultdict, deque
import time
CHAT_LIMIT = 12
CHAT_WINDOW = 60
chat_requests = defaultdict(deque)

@asynccontextmanager
async def lifespan(app):
    R.load_initial()
    sched.add_job(R.refresh, "interval", minutes=C.REFRESH_INTERVAL_MIN, max_instances=1, coalesce=True)
    sched.start()
    yield
    sched.shutdown(wait=False)


app = FastAPI(title="DarGlobal & Wasalt Assistant", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[dict] = []


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    if R.is_stale():                 # keep-alive ping doubles as a safety net for missed refreshes
        R.refresh_async()
    s = R.status()
    return {"status": "ok", "last_checked": s["last_checked"], "chunks": s["chunks"]}


@app.get("/status")
def status():
    return R.status()


@app.post("/refresh")
def manual_refresh(x_refresh_token: str = Header(default="")):
    if not C.REFRESH_TOKEN or x_refresh_token != C.REFRESH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")
    return R.refresh(force=True)


from fastapi import Request

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
