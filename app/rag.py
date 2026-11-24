import logging
import re
import time
from urllib.parse import urlsplit, urlunsplit

import requests

from . import config as C

log = logging.getLogger("rag")

API = "https://openrouter.ai/api/v1/chat/completions"
REQ_TIMEOUT = (10, 45)
NO_INFO_MSG = "I don't have that information in my data about DarGlobal and Wasalt."

_SESSION = requests.Session()

INSTRUCTIONS = """You are the assistant for DarGlobal (international luxury real estate developer) and Wasalt (Saudi real estate platform). Rules:
- Answer ONLY using the CONTEXT below. If the answer is not in the context, say: "I don't have that information in my data about DarGlobal and Wasalt." Do not guess or invent facts, numbers or names.
- Be clear and concise. Keep answers under about 250 words. Use short paragraphs or bullets when listing.
- If the user only greets you, greet back briefly and say what you can help with.
- If the question concerns both companies, cover both when the context allows.
- Stay strictly on the scope of the user's question (for example, when asked about locations or developments outside the UK, do not bring in UK cities unless directly asked).
- Treat marketing claims (rankings, projections, tax or investment benefits) as the company's own statements and attribute them, for example "According to DarGlobal's website, ...". Do not present them as independent facts.
- After each factual statement, add the number of the context block it came from in square brackets, like [2]. Never invent numbers."""

_FOLLOWUP_RE = re.compile(
    r"\b(it|its|they|them|their|theirs|those|these|there|this|that|same|former|latter)\b",
    re.I,
)

_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*(?:,|;|and)\s*\d+)*)\]", re.I)


def _norm_url(raw_url: str) -> str:
    parts = urlsplit((raw_url or "").strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _prompt(question: str, history: list, hits: list) -> str:
    ctx = "\n\n".join(
        f"[{i + 1}] ({c.get('title') or ''} | {c.get('url') or ''})\n{c.get('text') or ''}"
        for i, (c, _) in enumerate(hits)
    )
    convo = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}"
        for m in history[-4:]
    )
    convo_block = f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else ""
    return f"{INSTRUCTIONS}\n\nCONTEXT:\n{ctx}\n\n{convo_block}USER QUESTION: {question}\n\nANSWER:"


def _call_llm(prompt: str) -> str | None:
    if not C.OPENROUTER_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {C.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "DarGlobal-Wasalt Assistant",
    }
    models = [m for m in [C.MODEL_PRIMARY, *C.MODEL_FALLBACKS] if m]
    for model in models:
        for attempt in range(2):
            try:
                r = _SESSION.post(
                    API,
                    headers=headers,
                    timeout=REQ_TIMEOUT,
                    json={
                        "model": model,
                        "temperature": 0.2,
                        "max_tokens": 1200,
                        # Single user message: some free models reject the system role
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                    if out and out.strip():
                        return out.strip()
                    break
                log.warning("model %s -> HTTP %s", model, r.status_code)
                if r.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            except requests.RequestException as e:
                log.warning("model %s error: %s", model, e)
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                break
    return None


def _build_query(question: str, history: list) -> str:
    q = (question or "").strip()
    if not history or len(q.split()) >= 10:
        return q

    is_followup = len(q.split()) < 8 or bool(_FOLLOWUP_RE.search(q))
    if not is_followup:
        return q

    prev_user = next(
        (
            str(m.get("content", "")).strip()
            for m in reversed(history)
            if m.get("role") == "user" and str(m.get("content", "")).strip()
        ),
        "",
    )
    prev_assistant = next(
        (
            str(m.get("content", "")).strip()
            for m in reversed(history)
            if m.get("role") == "assistant" and str(m.get("content", "")).strip()
        ),
        "",
    )

    entity_hints: list[str] = []
    if prev_assistant:
        candidates = re.findall(
            r"\b[A-Z][a-zA-Z0-9&-]{2,}(?:\s+[A-Z][a-zA-Z0-9&-]{2,}){0,2}\b",
            prev_assistant,
        )
        seen: set[str] = set()
        for term in candidates:
            low = term.lower()
            if low not in {"according", "darglobal", "wasalt", "assistant", "information"} and low not in seen:
                seen.add(low)
                entity_hints.append(term)
            if len(entity_hints) >= 6:
                break

    parts = [p for p in (prev_user[:180], " ".join(entity_hints), q) if p]
    return " ".join(parts).strip()


def _extract_citations(text: str, max_idx: int) -> list[int]:
    cited: list[int] = []
    for group in _CITATION_GROUP_RE.findall(text):
        for raw_num in re.split(r"\s*(?:,|;|and)\s*", group, flags=re.I):
            if not raw_num.isdigit():
                continue
            n = int(raw_num)
            if 1 <= n <= max_idx and n not in cited:
                cited.append(n)
    return cited


def answer(question: str, history: list, index) -> dict:
    query = _build_query(question, history or [])
    hits = index.search(query)
    if not hits:
        return {"answer": NO_INFO_MSG, "sources": []}

    text = _call_llm(_prompt(question, history or [], hits))
    if text is None:
        return {
            "answer": "The AI model is busy or unavailable right now. Please try again in a moment.",
            "sources": [],
        }

    cited = _extract_citations(text, len(hits))
    clean = re.sub(
        r"\s+\[(?:\d+(?:\s*(?:,|;|and)\s*\d+)*)\](?=[.,;:!?)\]])",
        "",
        text,
        flags=re.I,
    )
    clean = _CITATION_GROUP_RE.sub("", clean)
    clean = re.sub(r"\s+([.,;:!?)\]])", r"\1", clean).strip()

    normalized_no_info = NO_INFO_MSG.lower()
    if clean.replace("’", "'").lower().startswith("i don't have that information"):
        return {"answer": clean or NO_INFO_MSG, "sources": []}
    if clean.lower() == normalized_no_info:
        return {"answer": NO_INFO_MSG, "sources": []}

    chosen = [hits[n - 1][0] for n in cited] or [c for c, _ in hits[:3]]
    seen_urls: set[str] = set()
    sources: list[dict] = []

    for c in chosen:
        raw_url = (c.get("url") or "").strip()
        if not raw_url:
            continue
        norm_url = _norm_url(raw_url)
        if norm_url in seen_urls:
            continue
        seen_urls.add(norm_url)
        title = (c.get("title") or "").strip() or raw_url
        sources.append({"title": title, "url": raw_url})
        if len(sources) >= 5:
            break

    return {"answer": clean, "sources": sources}