import logging, threading, time
from datetime import datetime, timezone

from . import config as C, indexer, scraper, store

log = logging.getLogger("refresh")
LOCK = threading.Lock()
INDEX = None          # live index; replaced atomically
STATE = {"running": False, "last_checked": None, "last_changed": None, "last_result": "not run yet",
         "report": {}, "_checked_ts": 0.0}


def _iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_initial():
    global INDEX
    store.ensure_runtime()
    pages = store.load_pages()
    if pages:
        INDEX = indexer.build(pages)
        STATE["last_changed"] = STATE["last_checked"] = max(r["fetched_at"] for r in pages.values())
        STATE["last_result"] = "loaded snapshot"


def is_stale() -> bool:
    return time.time() - STATE["_checked_ts"] > C.REFRESH_INTERVAL_MIN * 60


def refresh(force: bool = False) -> dict:
    global INDEX
    if not LOCK.acquire(blocking=False):
        return {"skipped": "refresh already running"}
    STATE["running"] = True
    try:
        old = store.load_pages()
        new, report = scraper.scrape_all(old)
        changed = [u for u in new if u not in old or new[u]["hash"] != old[u]["hash"]] + [u for u in old if u not in new]
        if new:
            store.save_pages(new)
        if new and (changed or force or INDEX is None):
            INDEX = indexer.build(new)          # built fully, then swapped in one assignment
            STATE["last_changed"] = _iso()
        STATE.update(last_checked=_iso(), _checked_ts=time.time(), report=report,
                     last_result=f"ok: {len(changed)} page(s) changed, {len(new)} total")
    except Exception as e:                       # the old index keeps serving
        log.exception("refresh failed")
        STATE["last_result"] = f"error: {e}"
    finally:
        STATE["running"] = False
        LOCK.release()
    return {"result": STATE["last_result"]}


def refresh_async():
    if not STATE["running"]:
        threading.Thread(target=refresh, daemon=True).start()


def status() -> dict:
    sites = {}
    if INDEX:
        for c in INDEX.chunks:
            sites[c["site"]] = sites.get(c["site"], 0) + 1
    pub = {k: v for k, v in STATE.items() if not k.startswith("_")}
    return {**pub, "chunks": len(INDEX.chunks) if INDEX else 0, "chunks_by_site": sites,
            "refresh_interval_min": C.REFRESH_INTERVAL_MIN}
