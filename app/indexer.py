"""Chunking, local embeddings (fastembed/ONNX) with a cache, and a compact hybrid (semantic + BM25) index."""
import hashlib, math, pickle, re, threading
from array import array
from collections import Counter

import numpy as np

from . import config as C

_model = None
_model_lock = threading.Lock()

STOP = set("a an the of and or to in on for with is are be what which who how do does can i my me you your "
           "it its at by from that this these those as about any there their have has all".split())
_tok_re = re.compile(r"[a-z0-9]+")


def _tok(s: str):
    return [t for t in _tok_re.findall(s.lower()) if len(t) < 40]


def _embedder():
    """Loaded lazily: with a shipped embedding cache, the model is only needed for the first chat question."""
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=C.EMBED_MODEL, cache_dir=str(C.MODEL_CACHE), threads=1)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    arr = np.array(list(_embedder().embed(texts, batch_size=C.EMBED_BATCH)), dtype="float32")
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
    """Semantic matrix + inverted keyword index (compact int arrays instead of a Counter per chunk)."""

    def __init__(self, chunks, mat):
        self.chunks, self.mat = chunks, mat
        n = len(chunks)
        dl = np.zeros(n, dtype="float32")
        post: dict[str, tuple[array, array]] = {}
        for i, c in enumerate(chunks):
            toks = _tok(c["title"] + " " + c["text"])
            dl[i] = len(toks)
            for t, f in Counter(toks).items():
                ids, tfs = post.setdefault(t, (array("i"), array("f")))
                ids.append(i)
                tfs.append(f)
        self.dl = dl
        self.avg = float(dl.mean()) if n else 1.0
        self.post, self.idf = {}, {}
        for t, (ids, tfs) in post.items():
            self.post[t] = (np.frombuffer(ids, dtype=np.int32), np.frombuffer(tfs, dtype=np.float32))
            self.idf[t] = math.log(1 + (n - len(ids) + .5) / (len(ids) + .5))

    def _bm25(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.chunks), dtype="float32")
        for t in {x for x in _tok(query) if x not in STOP}:
            p = self.post.get(t)
            if p is None:
                continue
            ids, tf = p
            norm = 1.2 * (0.25 + 0.75 * self.dl[ids] / self.avg)
            s[ids] += self.idf[t] * tf * 2.2 / (tf + norm)
        return s

    def search(self, query: str, k: int = C.TOP_K):
        if not self.chunks:
            return []
        sem = self.mat @ embed([query])[0]
        score = np.zeros(len(self.chunks))
        for arr in (sem, self._bm25(query)):          # reciprocal rank fusion of both rankings
            for rank, i in enumerate(np.argsort(-arr)[:40]):
                score[i] += 1 / (60 + rank)
        return [(self.chunks[i], float(score[i])) for i in np.argsort(-score)[:k]]


def _cache_path():
    return C.DATA_DIR / "emb_cache.pkl"


def _load_cache() -> dict:
    p = _cache_path()
    try:
        return pickle.loads(p.read_bytes()) if p.exists() else {}
    except Exception:
        return {}


def build(pages: dict) -> Index:
    """Chunk every page; embed only chunks missing from the cache, in small batches."""
    chunks = []
    for rec in pages.values():
        for part in chunk_text(rec["text"]):
            key = hashlib.sha256((rec["title"] + part).encode()).hexdigest()
            chunks.append({"url": rec["url"], "site": rec["site"], "title": rec["title"], "text": part, "key": key})
    cache = _load_cache()
    todo = list({c["key"]: c for c in chunks if c["key"] not in cache}.values())
    step = max(C.EMBED_BATCH * 4, 8)
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        vecs = embed([f'{c["title"]}\n{c["text"]}' for c in batch])
        for c, v in zip(batch, vecs):
            cache[c["key"]] = v
    keep = {c["key"]: cache[c["key"]] for c in chunks}
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_bytes(pickle.dumps(keep))
    mat = np.vstack([keep[c["key"]] for c in chunks]) if chunks else np.zeros((0, 384), dtype="float32")
    return Index(chunks, mat)
