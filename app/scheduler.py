from datetime import datetime, timezone
import gc
import threading
import time
from typing import Any, Callable, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchRefreshScheduler:
    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        gap_min: int,
        refresh_interval_min: int,
        run_batch_fn: Callable[[int, int], Dict[str, Any]],
        initial_chunks: int = 0,
        initial_chunks_by_site: Optional[Dict[str, int]] = None,
    ) -> None:
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.gap_sec = gap_min * 60
        self.refresh_interval_sec = refresh_interval_min * 60
        self.run_batch_fn = run_batch_fn
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        now_iso = _utc_now_iso()
        self.state: Dict[str, Any] = {
            "running": False,
            "mode": "batched",
            "batch_size": batch_size,
            "num_batches": num_batches,
            "batch_gap_min": gap_min,
            "batch_refresh_interval_min": refresh_interval_min,
            "active_batch": None,
            "last_batch_index": None,
            "last_checked": now_iso,
            "last_changed": now_iso,
            "last_result": "loaded snapshot",
            "report": {},
            "chunks": initial_chunks,
            "chunks_by_site": dict(initial_chunks_by_site or {}),
            "batches": [
                {
                    "index": i + 1,
                    "status": "pending",
                    "last_refreshed_ts": 0.0,
                    "updated_at": None,
                    "pages": 0,
                    "chunks": 0,
                }
                for i in range(num_batches)
            ],
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _next_due_batch_index(self) -> Optional[int]:
        now = time.time()
        for b in self.state["batches"]:
            if now - b["last_refreshed_ts"] >= self.refresh_interval_sec:
                return b["index"]
        return None

    def trigger_batch(self, batch_index: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if self.state["running"]:
                return {"started": False, "reason": "batch already running"}
            target = batch_index if batch_index is not None else self._next_due_batch_index()
            if target is None:
                return {"started": False, "reason": "no batch due"}
            if target < 1 or target > self.num_batches:
                return {"started": False, "reason": "invalid batch_index"}
            self.state["running"] = True
            self.state["active_batch"] = target
            self.state["batches"][target - 1]["status"] = "running"

        worker = threading.Thread(target=self._execute_batch, args=(target,), daemon=True)
        worker.start()
        return {"started": True, "batch_index": target}

    def _execute_batch(self, batch_index: int) -> None:
        try:
            summary = self.run_batch_fn(batch_index, self.batch_size)
            finished_ts = time.time()
            finished_iso = _utc_now_iso()
            with self._lock:
                b = self.state["batches"][batch_index - 1]
                b["status"] = "completed"
                b["last_refreshed_ts"] = finished_ts
                b["updated_at"] = finished_iso
                b["pages"] = summary.get("pages", 0)
                b["chunks"] = summary.get("chunks", 0)

                self.state["last_batch_index"] = batch_index
                self.state["last_checked"] = finished_iso
                if summary.get("changed", True):
                    self.state["last_changed"] = finished_iso
                if "total_chunks" in summary:
                    self.state["chunks"] = summary["total_chunks"]
                if "chunks_by_site" in summary:
                    self.state["chunks_by_site"] = dict(summary["chunks_by_site"])
                self.state["last_result"] = f"batch {batch_index}/{self.num_batches} completed"
                self.state["report"] = summary
        except Exception as exc:
            err_iso = _utc_now_iso()
            with self._lock:
                b = self.state["batches"][batch_index - 1]
                b["status"] = "error"
                self.state["last_checked"] = err_iso
                self.state["last_result"] = f"error on batch {batch_index}: {exc}"
        finally:
            gc.collect()
            with self._lock:
                self.state["running"] = False
                self.state["active_batch"] = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.trigger_batch(None)
            if self._stop_event.wait(self.gap_sec):
                break

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self.state,
                "chunks_by_site": dict(self.state["chunks_by_site"]),
                "batches": [dict(b) for b in self.state["batches"]],
            }