import json, os, shutil
from pathlib import Path
from . import config as C


def pages_path() -> Path:
    return C.DATA_DIR / "pages.json"


def ensure_runtime():
    """First boot: copy the committed snapshot and its embeddings, so nothing has to be embedded at startup."""
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("pages.json", "emb_cache.pkl"):
        src, dst = C.SEED_DIR / name, C.DATA_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)


def load_pages(path: Path | None = None) -> dict:
    p = path or pages_path()
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def save_pages(pages: dict, path: Path | None = None):
    p = path or pages_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(pages, ensure_ascii=False), "utf-8")
    os.replace(tmp, p)
