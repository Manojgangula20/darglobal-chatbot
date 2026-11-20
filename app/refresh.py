from datetime import datetime, timezone
import gc
import logging
import threading
import time

from . import config as C, indexer, scraper, store
from .nlp_processor import chunk_text, lemmatize_text, normalize_text

log = logging.getLogger("refresh")
LOCK = threading.Lock()
INDEX = None

BATCH_SIZE = getattr(C, "BATCH_SIZE", 5)
NUM_BATCHES = getattr(C, "NUM_BATCHES", 5)
BATCH_GAP_MIN = getattr(C, "BATCH_GAP_MIN", 60)
BATCH_REFRESH_INTERVAL_MIN = getattr(C, "BATCH_REFRESH_INTERVAL_MIN", 360)

STATE = {
    "running": False,
    "mode": "batched",
    "batch_size": BATCH_SIZE,
    "num_batches": NUM_BATCHES,
    "batch_gap_min": BATCH_GAP_MIN,
    "batch_refresh_interval_min": BATCH_REFRESH_INTERVAL_MIN,
    "active_batch": None,
    "last_batch_index": None,
    "last_checked": None,
    "last_changed": None,
    "last_result": "not run yet",
    "report": {},
    "_checked_ts": 0.0,
    "_batches": [
        {
            "index": i + 1,
            "status": "pending",
            "last_refreshed_ts": 0.0,
            "updated_at": None,
            "pages": 0,
            "chunks": 0,
        }
        for i in range(NUM_BATCHES)
    ],
}


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_batch_pages(pages: dict, batch_index: int, batch_size: int) -> dict:
    urls = sorted(pages.keys())
    start = (batch_index - 1) * batch_size
    end = start + batch_size
    selected = urls[start:end]
    batch_pages: dict = {}

    for u in selected:
        rec = dict(pages[u])
        raw_text = rec.get("text") or rec.get("content") or rec.get("body") or ""
        norm_text = normalize_text(raw_text)
        if norm_text:
            rec["text"] = norm_text
            rec["lemma_text"] = lemmatize_text(norm_text)
            rec["chunks"] = chunk_text(
                norm_text,
                max_chars=getattr(C, "CHUNK_MAX_CHARS", 900),
                overlap=getattr(C, "CHUNK_OVERLAP", 120),
            )
        rec["fetched_at"] = _iso()
        batch_pages[u] = rec

    return batch_pages


def _pick_due_batch() -> int | None:
    now = time.time()
    interval_sec = BATCH_REFRESH_INTERVAL_MIN * 60
    for b in STATE["_batches"]:
        if now - b["last_refreshed_ts"] >= interval_sec:
            return b["index"]
    return None


def load_initial():
    global INDEX
    store.ensure_runtime()
    pages = store.load_pages()
    if pages:
        for rec in pages.values():
            txt = rec.get("text") or rec.get("content") or ""
            if txt:
                norm = normalize_text(txt)
                rec["text"] = norm
                rec.setdefault("lemma_text", lemmatize_text(norm))
        INDEX = indexer.build(pages)
        STATE["last_changed"] = STATE["last_checked"] = max(r["fetched_at"] for r in pages.values())
        STATE["last_result"] = "loaded snapshot"
        STATE["_checked_ts"] = datetime.fromisoformat(STATE["last_checked"]).timestamp()
    del pages
    gc.collect()


def is_stale() -> bool:
    return _pick_due_batch() is not None


def get_batch_targets(batch_index: int, batch_size: int) -> list[tuple[str, str]]:
    pages = store.load_pages() or {}
    urls = sorted(pages.keys())
    start = (batch_index - 1) * batch_size
    end = start + batch_size
    out: list[tuple[str, str]] = []
    for u in urls[start:end]:
        site = pages[u].get("site", "darglobal")
        out.append((site, u))
    return out


def scrape_single_page(site: str, url: str) -> str:
    pages = store.load_pages() or {}
    rec = pages.get(url) or {}
    return rec.get("text") or rec.get("content") or ""


def upsert_batch_chunks(prepared_chunks: list[dict]) -> int:
    global INDEX
    if not prepared_chunks:
        return 0
    all_pages = store.load_pages() or {}
    by_url: dict[str, list[dict]] = {}
    for ch in prepared_chunks:
        by_url.setdefault(ch["url"], []).append(ch)

    for url, ch_list in by_url.items():
        rec = dict(all_pages.get(url, {"site": ch_list[0]["site"], "url": url}))
        merged_text = " ".join(c["text"] for c in ch_list)
        rec["text"] = merged_text
        rec["lemma_text"] = " ".join(c.get("lemma_text", "") for c in ch_list)
        rec["fetched_at"] = _iso()
        all_pages[url] = rec

    store.save_pages(all_pages)
    INDEX = indexer.build(all_pages)
    return len(prepared_chunks)


def refresh(force: bool = False, batch_index: int | None = None) -> dict:
    global INDEX
    if not LOCK.acquire(blocking=False):
        return {"skipped": "refresh already running"}

    target = batch_index if batch_index is not None else _pick_due_batch()
    if target is None and force:
        last_idx = STATE.get("last_batch_index") or 0
        target = (last_idx % NUM_BATCHES) + 1
    if target is None:
        LOCK.release()
        return {"skipped": "no batch due"}
    if target < 1 or target > NUM_BATCHES:
        LOCK.release()
        return {"skipped": "invalid batch_index"}

    STATE["running"] = True
    STATE["active_batch"] = target
    STATE["_batches"][target - 1]["status"] = "running"

    try:
        all_pages = store.load_pages() or {}
        batch_updates = _build_batch_pages(all_pages, target, BATCH_SIZE)
        if batch_updates:
            for u, rec in batch_updates.items():
                all_pages[u] = rec
            store.save_pages(all_pages)
            INDEX = indexer.build(all_pages)
            now_iso = _iso()
            STATE["last_changed"] = now_iso
            STATE["_batches"][target - 1]["updated_at"] = now_iso

        finished_ts = time.time()
        finished_iso = _iso()
        chunk_count = sum(len(r.get("chunks", [])) for r in batch_updates.values())

        STATE["_batches"][target - 1].update(
            status="completed",
            last_refreshed_ts=finished_ts,
            updated_at=finished_iso,
            pages=len(batch_updates),
            chunks=chunk_count,
        )
        STATE.update(
            last_batch_index=target,
            last_checked=finished_iso,
            _checked_ts=finished_ts,
            last_result=f"batch {target}/{NUM_BATCHES} completed ({len(batch_updates)} pages)",
            report={"batch_index": target, "pages": len(batch_updates), "chunks": chunk_count},
        )
    except Exception as e:
        log.exception("batch refresh failed")
        err_iso = _iso()
        STATE["_batches"][target - 1]["status"] = "error"
        STATE.update(last_checked=err_iso, last_result=f"error: {e}")
    finally:
        STATE["running"] = False
        STATE["active_batch"] = None
        gc.collect()
        LOCK.release()

    return {"result": STATE["last_result"], "batch_index": target}


def refresh_async(batch_index: int | None = None):
    if not STATE["running"]:
        threading.Thread(target=refresh, kwargs={"batch_index": batch_index}, daemon=True).start()


def status() -> dict:
    sites = {}
    if INDEX:
        for c in INDEX.chunks:
            sites[c["site"]] = sites.get(c["site"], 0) + 1
    pub = {k: v for k, v in STATE.items() if not k.startswith("_")}
    pub["batches"] = [dict(b) for b in STATE["_batches"]]
    return {
        **pub,
        "chunks": len(INDEX.chunks) if INDEX else 0,
        "chunks_by_site": sites,
        "refresh_interval_min": BATCH_REFRESH_INTERVAL_MIN,
    }