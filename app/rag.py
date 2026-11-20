import logging, re, time
from urllib.parse import urlsplit, urlunsplit
import requests
from . import config as C

log = logging.getLogger("rag")
API = "https://openrouter.ai/api/v1/chat/completions"

INSTRUCTIONS = """You are the assistant for DarGlobal (international luxury real estate developer) and Wasalt (Saudi real estate platform).
Rules:
- Answer ONLY using the CONTEXT below. If the answer is not in the context, say: "I don't have that information in my data about DarGlobal and Wasalt." Do not guess or invent facts, numbers or names.
- Be clear and concise. Keep answers under about 250 words. Use short paragraphs or bullets when listing.
- If the user only greets you, greet back briefly and say what you can help with.
- If the question concerns both companies, cover both when the context allows.
- Treat marketing claims (rankings, projections, tax or investment benefits) as the company's own statements and attribute them, for example "According to DarGlobal's website, ...". Do not present them as independent facts.
- After each factual statement, add the number of the context block it came from in square brackets, like [2]. Never invent numbers.
"""


def _norm_url(raw_url: str) -> str:
    parts = urlsplit((raw_url or "").strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _prompt(question, history, hits):
    ctx = "\n\n".join(f"[{i + 1}] ({c['title']} | {c['url']})\n{c['text']}" for i, (c, _) in enumerate(hits))
    convo = "\n".join(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}" for m in history[-4:])
    return f"{INSTRUCTIONS}\n\nCONTEXT:\n{ctx}\n\n" + (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "") + \
        f"USER QUESTION: {question}\n\nANSWER:"


def _call_llm(prompt: str) -> str | None:
    if not C.OPENROUTER_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {C.OPENROUTER_API_KEY}", "X-Title": "DarGlobal-Wasalt Assistant"}
    for model in [C.MODEL_PRIMARY, *C.MODEL_FALLBACKS]:
        for attempt in range(2):
            try:
                r = requests.post(API, headers=headers, timeout=60, json={
                    "model": model,
                    "temperature": 0.2,
                    "max_tokens": 1200,
                    # single user message: some free models reject the system role
                    "messages": [{"role": "user", "content": prompt}]
                })
                if r.status_code == 200:
                    out = (r.json().get("choices") or [{}])[0].get("message", {}).get("content")
                    if out and out.strip():
                        return out.strip()
                log.warning("model %s -> HTTP %s", model, r.status_code)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.5)
                    continue
                break
            except requests.RequestException as e:
                log.warning("model %s error: %s", model, e)
    return None


def answer(question: str, history: list, index) -> dict:
    prev = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    query = f"{prev[:300]} {question}" if prev and len(question.split()) < 8 else question
    hits = index.search(query)
    text = _call_llm(_prompt(question, history, hits))
    if text is None:
        return {"answer": "The AI model is busy or unavailable right now. Please try again in a moment.", "sources": []}

    cited = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        for n in (int(x) for x in group.split(",")):
            if 1 <= n <= len(hits) and n not in cited:
                cited.append(n)

    clean = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text).strip()
    if clean.replace("’", "'").lower().startswith("i don't have that information"):
        return {"answer": clean, "sources": []}

    chosen = [hits[n - 1][0] for n in cited] or [c for c, _ in hits[:3]]
    seen_urls, seen_titles, sources = set(), set(), []
    for c in chosen:
        raw_url = (c.get("url") or "").strip()
        if not raw_url:
            continue
        norm_url = _norm_url(raw_url)
        title = (c.get("title") or "").strip() or raw_url
        norm_title = title.lower()
        if norm_url not in seen_urls and norm_title not in seen_titles and len(sources) < 4:
            seen_urls.add(norm_url)
            seen_titles.add(norm_title)
            sources.append({"title": title, "url": raw_url})

    return {"answer": clean, "sources": sources}
