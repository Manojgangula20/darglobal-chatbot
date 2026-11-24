"""Chunking, local embeddings (fastembed/ONNX), and Qdrant-backed hybrid (semantic + BM25) search."""

from array import array
from collections import Counter
import gc
import hashlib
import logging
import math
import threading
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from . import config as C
from .nlp_processor import chunk_text as _nlp_chunk_text, lemmatize_text, normalize_text

log = logging.getLogger("indexer")

_model = None
_model_lock = threading.Lock()
_qdrant = None
_qdrant_lock = threading.Lock()
_indexes_ready = False
_indexes_lock = threading.Lock()

VECTOR_SIZE = 384
_POINT_NS = uuid.UUID("7f3b8d9a-4c2e-4f1a-9b8d-1e2f3a4b5c6d")

STOP = set(
    "a an the of and or to in on for with is are be what which who how do does can i my me you your "
    "it its at by from that this these those as about any there their have has all".split()
)


def _tok(s: str) -> list[str]:
    lem = lemmatize_text(s or "")
    return [t for t in lem.split() if t and len(t) < 40]


def _point_id(key_hex: str) -> str:
    return str(uuid.uuid5(_POINT_NS, key_hex))


def _payload_key(title: str, text: str, payload: dict | None = None) -> str:
    if payload and payload.get("key"):
        return str(payload["key"])
    return hashlib.sha256((title + text).encode("utf-8")).hexdigest()


def get_qdrant() -> QdrantClient:
    global _qdrant
    with _qdrant_lock:
        if _qdrant is None:
            if not C.QDRANT_URL or not C.QDRANT_API_KEY:
                raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be configured")
            _qdrant = QdrantClient(
                url=C.QDRANT_URL,
                api_key=C.QDRANT_API_KEY,
                prefer_grpc=False,
                timeout=30.0,
            )
        return _qdrant


def _ensure_payload_indexes(client: QdrantClient):
    global _indexes_ready
    if _indexes_ready:
        return
    with _indexes_lock:
        if _indexes_ready:
            return
        for field_name, schema in (
            ("site", PayloadSchemaType.KEYWORD),
            ("url", PayloadSchemaType.KEYWORD),
            ("text", PayloadSchemaType.TEXT),
            ("title", PayloadSchemaType.TEXT),
        ):
            try:
                client.create_payload_index(
                    collection_name=C.QDRANT_COLLECTION,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:
                log.debug("payload index check skipped/existing for %s: %s", field_name, exc)
        _indexes_ready = True


def ensure_collection():
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if C.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=C.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info("created Qdrant collection: %s", C.QDRANT_COLLECTION)
    _ensure_payload_indexes(client)


def collection_counts() -> tuple[int, dict[str, int]]:
    client = get_qdrant()
    total = client.count(collection_name=C.QDRANT_COLLECTION, exact=True).count
    by_site: dict[str, int] = {}
    for site in ("darglobal", "wasalt", "wasalt-blog"):
        cnt = client.count(
            collection_name=C.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="site", match=MatchValue(value=site))]
            ),
            exact=True,
        ).count
        if cnt:
            by_site[site] = cnt
    return total, by_site


def _embedder():
    """Loaded lazily only when embedding new chunks or incoming chat queries."""
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=C.EMBED_MODEL, cache_dir=str(C.MODEL_CACHE), threads=1)
        return _model


def embed(texts: list[str]) -> np.ndarray:
    arr = np.array(list(_embedder().embed(texts, batch_size=C.EMBED_BATCH)), dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return (arr / norms).astype("float32", copy=False)


def chunk_text(
    text: str,
    max_chars: int = C.CHUNK_MAX_CHARS,
    overlap: int = C.CHUNK_OVERLAP,
) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    return _nlp_chunk_text(norm, max_chars=max_chars, overlap=overlap)


def _bm25_rerank(query: str, candidates: list[dict], doc_tokens: list[list[str]] | None = None) -> np.ndarray:
    n = len(candidates)
    if n == 0:
        return np.zeros(0, dtype="float32")
    dl = np.zeros(n, dtype="float32")
    post: dict[str, tuple[array, array]] = {}
    for i, c in enumerate(candidates):
        toks = doc_tokens[i] if doc_tokens is not None else _tok(f"{c.get('title', '')} {c.get('text', '')}")
        dl[i] = len(toks)
        for t, f in Counter(toks).items():
            ids, tfs = post.setdefault(t, (array("i"), array("f")))
            ids.append(i)
            tfs.append(f)
    avg = float(dl.mean()) if n else 1.0
    if avg <= 0:
        avg = 1.0
    s = np.zeros(n, dtype="float32")
    for t in {x for x in _tok(query) if x not in STOP}:
        p = post.get(t)
        if p is None:
            continue
        ids_arr, tf_arr = p
        ids = np.frombuffer(ids_arr, dtype=np.int32)
        tf = np.frombuffer(tf_arr, dtype=np.float32)
        idf = math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
        norm = 1.2 * (0.25 + 0.75 * dl[ids] / avg)
        s[ids] += idf * tf * 2.2 / (tf + norm)
    return s


def _keyword_candidates(client: QdrantClient, query: str, limit: int = 20) -> list[dict]:
    """Fetch extra candidates matching distinctive query terms so exact names are not missed."""
    terms = sorted(
        {t for t in _tok(query) if t not in STOP and len(t) >= 3},
        key=len,
        reverse=True,
    )[:3]
    if not terms:
        return []
    _ensure_payload_indexes(client)
    found: list[dict] = []
    seen_keys: set[str] = set()
    for term in terms:
        try:
            pts, _ = client.scroll(
                collection_name=C.QDRANT_COLLECTION,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(key="text", match=MatchText(text=term)),
                        FieldCondition(key="title", match=MatchText(text=term)),
                    ]
                ),
                limit=max(4, limit // len(terms)),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            log.warning("keyword scroll failed for term=%r: %s", term, exc)
            continue
        for p in pts or []:
            pl = p.payload or {}
            title = pl.get("title", "")
            text = pl.get("text", "")
            key = _payload_key(title, text, pl)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            found.append(
                {
                    "url": pl.get("url", ""),
                    "site": pl.get("site", ""),
                    "title": title,
                    "text": text,
                    "key": key,
                }
            )
            if len(found) >= limit:
                return found
    return found


class QdrantIndex:
    """Zero-RAM-storage index querying Qdrant Cloud + BM25 reciprocal rank fusion."""

    def __init__(self, initial_count: int = 0):
        self._count = initial_count

    @property
    def has_data(self) -> bool:
        return self._count > 0

    def set_count(self, count: int):
        self._count = count

    def search(self, query: str, k: int = C.TOP_K):
        client = get_qdrant()
        q_vec = embed([query])[0].tolist()
        resp = client.query_points(
            collection_name=C.QDRANT_COLLECTION,
            query=q_vec,
            limit=40,
            with_payload=True,
        )
        points = resp.points or []
        if not points:
            return []

        candidates: list[dict] = []
        key_to_idx: dict[str, int] = {}
        dense_ranks: dict[int, int] = {}

        for p in points:
            pl = p.payload or {}
            title = pl.get("title", "")
            text = pl.get("text", "")
            key = _payload_key(title, text, pl)
            if key in key_to_idx:
                continue
            idx = len(candidates)
            key_to_idx[key] = idx
            dense_ranks[idx] = len(dense_ranks)
            candidates.append(
                {
                    "url": pl.get("url", ""),
                    "site": pl.get("site", ""),
                    "title": title,
                    "text": text,
                    "key": key,
                }
            )

        for kw_cand in _keyword_candidates(client, query, limit=15):
            if kw_cand["key"] not in key_to_idx:
                key_to_idx[kw_cand["key"]] = len(candidates)
                candidates.append(kw_cand)

        n = len(candidates)
        score = np.zeros(n, dtype="float32")
        for cand_idx, rank in dense_ranks.items():
            score[cand_idx] += 1.0 / (60.0 + rank)

        doc_tokens = [_tok(f"{c.get('title', '')} {c.get('text', '')}") for c in candidates]
        bm25_scores = _bm25_rerank(query, candidates, doc_tokens=doc_tokens)
        for rank, cand_idx in enumerate(np.argsort(-bm25_scores)):
            score[cand_idx] += 1.0 / (60.0 + rank)

        return [(candidates[i], float(score[i])) for i in np.argsort(-score)[:k]]


def upsert_chunks(chunks: list[dict], batch_size: int = 64) -> int:
    """Embed and upsert a list of chunk dicts directly into Qdrant without holding a global index."""
    if not chunks:
        return 0
    ensure_collection()
    client = get_qdrant()

    dedup: dict[str, dict] = {}
    for c in chunks:
        title = c.get("title", "")
        text = c.get("text", "")
        key = _payload_key(title, text, c)
        dedup[key] = {
            "url": c.get("url", ""),
            "site": c.get("site", ""),
            "title": title,
            "text": text,
            "key": key,
        }

    items = list(dedup.values())
    step = max(C.EMBED_BATCH * 4, 16)
    uploaded = 0

    for i in range(0, len(items), step):
        batch = items[i : i + step]
        vecs = embed([f'{c["title"]}\n{c["text"]}' for c in batch])
        points = [
            PointStruct(
                id=_point_id(c["key"]),
                vector=v.tolist(),
                payload=c,
            )
            for c, v in zip(batch, vecs)
        ]
        for j in range(0, len(points), batch_size):
            client.upsert(
                collection_name=C.QDRANT_COLLECTION,
                points=points[j : j + batch_size],
                wait=True,
            )
        uploaded += len(batch)
        print(f"Uploaded {uploaded}/{len(items)} chunks to Qdrant...", flush=True)
        del vecs, points
        gc.collect()

    return uploaded


def build(pages: dict) -> QdrantIndex:
    """Chunk every page and upload all chunks to Qdrant Cloud."""
    ensure_collection()
    chunks = []
    for rec in pages.values():
        txt = normalize_text(rec.get("text") or "")
        if not txt:
            continue
        title = rec.get("title") or rec.get("url") or ""
        for part in chunk_text(txt):
            key = _payload_key(title, part)
            chunks.append(
                {
                    "url": rec.get("url", ""),
                    "site": rec.get("site", ""),
                    "title": title,
                    "text": part,
                    "key": key,
                }
            )

    total_uploaded = upsert_chunks(chunks)
    del chunks
    gc.collect()
    total, _ = collection_counts()
    log.info("Qdrant build complete: uploaded=%s total_in_qdrant=%s", total_uploaded, total)
    return QdrantIndex(initial_count=total)