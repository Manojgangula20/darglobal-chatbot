from datetime import datetime, timezone
import gc
import hashlib
import logging
import threading
import time

from . import config as C, indexer, scraper, store
from .nlp_processor import chunk_text, lemmatize_text, normalize_text

log = logging.getLogger("refresh")
LOCK = threading.Lock()
INDEX: indexer.QdrantIndex | None = None

BATCH_SIZE = getattr(C, "BATCH_SIZE", 10)
NUM_BATCHES = getattr(C, "NUM_BATCHES", 100)
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

_CACHED_COUNTS: tuple[int, dict[str, int]] = (0, {})


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _refresh_counts() -> tuple[int, dict[str, int]]:
    global _CACHED_COUNTS
    total, by_site = indexer.collection_counts()
    _CACHED_COUNTS = (total, by_site)
    if INDEX is not None:
        INDEX.set_count(total)
    return total, by_site


def _find_existing_page(pages: dict, site: str, url: str) -> dict | None:
    if url in pages:
        return pages[url]
    norm_u = scraper.norm(url)
    if norm_u in pages:
        return pages[norm_u]
    for rec in pages.values():
        if rec.get("url") == url or scraper.norm(rec.get("url", "")) == norm_u:
            if not site or rec.get("site") == site:
                return rec
    return None


def _build_batch_pages(pages: dict, batch_index: int, batch_size: int) -> dict:
    urls = sorted(pages.keys())
    start = (batch_index - 1) * batch_size
    end = start + batch_size
    selected = urls[start:end]
    batch_pages: dict = {}
    log.info("batch %s/%s building pages: %s targets", batch_index, NUM_BATCHES, len(selected))

    max_chars = getattr(C, "CHUNK_MAX_CHARS", 900)
    overlap = getattr(C, "CHUNK_OVERLAP", 120)

    for u in selected:
        existing = _find_existing_page(pages, pages[u].get("site", "darglobal"), u) or pages[u]
        site = existing.get("site") or pages[u].get("site") or "darglobal"
        target_url = existing.get("url") or u
        fetched = scraper.fetch_single_page(site, target_url, existing)
        rec = dict(fetched or existing)
        raw_text = rec.get("text") or rec.get("content") or rec.get("body") or ""
        norm_text = normalize_text(raw_text)
        if norm_text:
            rec["text"] = norm_text
            rec["lemma_text"] = lemmatize_text(norm_text)
            rec["chunks"] = chunk_text(norm_text, max_chars=max_chars, overlap=overlap)
            log.info(
                "batch %s prepared url=%s site=%s chunks=%s",
                batch_index,
                target_url,
                site,
                len(rec["chunks"]),
            )
        else:
            log.warning("batch %s empty normalized text for url=%s", batch_index, target_url)
        rec["site"] = site
        rec["url"] = target_url
        rec["fetched_at"] = _iso()
        batch_pages[u] = rec

    return batch_pages


def _pick_due_batch() -> int | None:
    now = time.time()
    interval_sec = BATCH_REFRESH_INTERVAL_MIN * 60
    total = len(STATE["_batches"])
    if total == 0:
        return None
    last_idx = STATE.get("last_batch_index") or 0
    start_offset = last_idx % total
    for step in range(total):
        b = STATE["_batches"][(start_offset + step) % total]
        if now - b["last_refreshed_ts"] >= interval_sec:
            return b["index"]
    return None


def load_initial():
    global INDEX
    store.ensure_runtime()
    indexer.ensure_collection()
    INDEX = indexer.QdrantIndex()
    total, by_site = _refresh_counts()
    if total == 0:
        pages = store.load_pages()
        if pages:
            log.info("Qdrant collection empty; seeding from local snapshot (%s pages)...", len(pages))
            INDEX = indexer.build(pages)
            total, by_site = _refresh_counts()
            del pages
            gc.collect()
    now_iso = _iso()
    STATE["last_changed"] = STATE["last_checked"] = now_iso
    STATE["last_result"] = f"connected to Qdrant ({total} chunks)"
    STATE["_checked_ts"] = time.time()
    log.info("connected to Qdrant Cloud: chunks=%s by_site=%s", total, by_site)
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
    existing = _find_existing_page(pages, site, url)
    fetched = scraper.fetch_single_page(site, url, existing)
    if fetched and fetched.get("text"):
        key = url if url in pages else (fetched.get("url") or url)
        pages[key] = {**(existing or {}), **fetched}
        store.save_pages(pages)
        return fetched["text"]
    return (existing or {}).get("text") or (existing or {}).get("content") or ""


def upsert_batch_chunks(prepared_chunks: list[dict]) -> int:
    if not prepared_chunks:
        return 0
    all_pages = store.load_pages() or {}
    by_url: dict[str, list[dict]] = {}
    for ch in prepared_chunks:
        by_url.setdefault(ch["url"], []).append(ch)
    qdrant_docs = []
    for url, ch_list in by_url.items():
        rec = dict(all_pages.get(url, {"site": ch_list[0]["site"], "url": url}))
        title = rec.get("title") or ch_list[0].get("title") or url
        merged_text = " ".join(c["text"] for c in ch_list)
        rec["title"] = title
        rec["text"] = merged_text
        rec["lemma_text"] = " ".join(c.get("lemma_text", "") for c in ch_list)
        rec["fetched_at"] = _iso()
        all_pages[url] = rec
        for c in ch_list:
            part = c["text"]
            qdrant_docs.append(
                {
                    "url": url,
                    "site": c.get("site", rec.get("site", "")),
                    "title": title,
                    "text": part,
                    "key": hashlib.sha256((title + part).encode("utf-8")).hexdigest(),
                }
            )
    store.save_pages(all_pages)
    del all_pages
    uploaded = indexer.upsert_chunks(qdrant_docs)
    _refresh_counts()
    gc.collect()
    log.info("upserted batch chunks to Qdrant: count=%s urls=%s", uploaded, len(by_url))
    return uploaded


def refresh(force: bool = False, batch_index: int | None = None) -> dict:
    if not LOCK.acquire(blocking=False):
        return {"skipped": "refresh already running"}
    try:
        target = batch_index if batch_index is not None else _pick_due_batch()
        if target is None and force:
            last_idx = STATE.get("last_batch_index") or 0
            target = (last_idx % NUM_BATCHES) + 1
        if target is None:
            return {"skipped": "no batch due"}
        if target < 1 or target > NUM_BATCHES:
            return {"skipped": "invalid batch_index"}

        STATE["running"] = True
        STATE["active_batch"] = target
        STATE["_batches"][target - 1]["status"] = "running"
        log.info("batch %s/%s refresh started (force=%s)", target, NUM_BATCHES, force)

        max_chars = getattr(C, "CHUNK_MAX_CHARS", 900)
        overlap = getattr(C, "CHUNK_OVERLAP", 120)

        try:
            all_pages = store.load_pages() or {}
            batch_updates = _build_batch_pages(all_pages, target, BATCH_SIZE)
            chunk_count = 0
            if batch_updates:
                qdrant_docs = []
                for u, rec in batch_updates.items():
                    all_pages[u] = rec
                    title = rec.get("title") or rec.get("url") or u
                    for part in chunk_text(rec.get("text", ""), max_chars=max_chars, overlap=overlap):
                        qdrant_docs.append(
                            {
                                "url": rec.get("url", u),
                                "site": rec.get("site", ""),
                                "title": title,
                                "text": part,
                                "key": hashlib.sha256((title + part).encode("utf-8")).hexdigest(),
                            }
                        )
                store.save_pages(all_pages)
                del all_pages
                chunk_count = indexer.upsert_chunks(qdrant_docs)
                _refresh_counts()
                now_iso = _iso()
                STATE["last_changed"] = now_iso
                STATE["_batches"][target - 1]["updated_at"] = now_iso
            else:
                del all_pages

            finished_ts = time.time()
            finished_iso = _iso()
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
            log.info(
                "batch %s/%s completed: pages=%s chunks=%s",
                target,
                NUM_BATCHES,
                len(batch_updates),
                chunk_count,
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

        return {"result": STATE["last_result"], "batch_index": target}
    finally:
        LOCK.release()


def refresh_async(batch_index: int | None = None):
    if not STATE["running"]:
        threading.Thread(target=refresh, kwargs={"batch_index": batch_index}, daemon=True).start()


def status() -> dict:
    total, sites = _CACHED_COUNTS
    pub = {k: v for k, v in STATE.items() if not k.startswith("_")}
    pub["batches"] = [dict(b) for b in STATE["_batches"]]
    return {**pub, "chunks": total, "chunks_by_site": sites, "refresh_interval_min": BATCH_REFRESH_INTERVAL_MIN}