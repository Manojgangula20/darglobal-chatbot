import json, os, shutil
from pathlib import Path
from . import config as C


def pages_path() -> Path:
    return C.DATA_DIR / "pages.json"


def ensure_runtime():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed = C.SEED_DIR / "pages.json"
    if not pages_path().exists() and seed.exists():
        shutil.copy(seed, pages_path())


def load_pages(path: Path | None = None) -> dict:
    p = path or pages_path()
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def save_pages(pages: dict, path: Path | None = None):
    p = path or pages_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(pages, ensure_ascii=False), "utf-8")
    os.replace(tmp, p)
