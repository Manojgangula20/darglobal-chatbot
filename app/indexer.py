"""Chunking, local embeddings (fastembed/ONNX) with a cache, and a small in-memory vector index."""
import hashlib, pickle, threading
import numpy as np

from . import config as C

_model = None
_model_lock = threading.Lock()


def _embedder():
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=C.EMBED_MODEL, cache_dir=str(C.MODEL_CACHE))
    return _model


def embed(texts: list[str]) -> np.ndarray:
    arr = np.array(list(_embedder().embed(texts)), dtype="float32")
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def chunk_text(text: str, max_chars: int = 1200, overlap_lines: int = 2) -> list[str]:
    chunks, cur, n = [], [], 0
    for ln in text.split("\n"):
        if n + len(ln) > max_chars and cur:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_lines:]
            n = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


class Index:
    def __init__(self, chunks, mat):
        self.chunks, self.mat = chunks, mat

    def search(self, query: str, k: int = C.TOP_K):
        if not self.chunks:
            return []
        sims = self.mat @ embed([query])[0]
        return [(self.chunks[i], float(sims[i])) for i in np.argsort(-sims)[:k]]


def _cache_path():
    return C.DATA_DIR / "emb_cache.pkl"


def _load_cache() -> dict:
    p = _cache_path()
    try:
        return pickle.loads(p.read_bytes()) if p.exists() else {}
    except Exception:
        return {}


def build(pages: dict) -> Index:
    """Chunk every page and embed only chunks not already cached (unchanged pages cost nothing)."""
    chunks = []
    for rec in pages.values():
        for part in chunk_text(rec["text"]):
            key = hashlib.sha256((rec["title"] + part).encode()).hexdigest()
            chunks.append({"url": rec["url"], "site": rec["site"], "title": rec["title"], "text": part, "key": key})
    cache = _load_cache()
    todo = list({c["key"]: c for c in chunks if c["key"] not in cache}.values())
    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        vecs = embed([f'{c["title"]}\n{c["text"]}' for c in batch])
        for c, v in zip(batch, vecs):
            cache[c["key"]] = v
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_bytes(pickle.dumps({c["key"]: cache[c["key"]] for c in chunks}))
    mat = np.vstack([cache[c["key"]] for c in chunks]) if chunks else np.zeros((0, 384), dtype="float32")
    return Index(chunks, mat)
