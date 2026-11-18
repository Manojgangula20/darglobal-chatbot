import logging, time
import requests
from . import config as C

log = logging.getLogger("rag")
API = "https://openrouter.ai/api/v1/chat/completions"

INSTRUCTIONS = """You are the assistant for DarGlobal (international luxury real estate developer) and Wasalt (Saudi real estate platform).
Rules:
- Answer ONLY using the CONTEXT below. If the answer is not in the context, say: "I don't have that information in my data about DarGlobal and Wasalt." Do not guess or invent facts, numbers or names.
- Be clear and concise. Use short paragraphs or bullets when listing.
- If the user only greets you, greet back briefly and say what you can help with.
- If the question concerns both companies, cover both when the context allows.
- Treat marketing claims (rankings, projections, tax or investment benefits) as the company's own statements and attribute them, for example "According to DarGlobal's website, ...". Do not present them as independent facts.
"""


def _prompt(question, history, hits):
    ctx = "\n\n".join(f"[{i + 1}] ({c['title']} | {c['url']})\n{c['text']}" for i, (c, _) in enumerate(hits))
    convo = "\n".join(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}"
                      for m in history[-4:])
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
                    "model": model, "temperature": 0.2, "max_tokens": 700,
                    # single user message: some free models reject the system role
                    "messages": [{"role": "user", "content": prompt}]})
                if r.status_code == 200:
                    out = (r.json().get("choices") or [{}])[0].get("message", {}).get("content")
                    if out and out.strip():
                        return out.strip()
                log.warning("model %s -> HTTP %s", model, r.status_code)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.5)
                    continue
                break                              # other 4xx: try the next model
            except requests.RequestException as e:
                log.warning("model %s error: %s", model, e)
    return None


def answer(question: str, history: list, index) -> dict:
    hits = index.search(question)
    text = _call_llm(_prompt(question, history, hits))
    if text is None:
        return {"answer": "The AI model is busy or unavailable right now. Please try again in a moment.",
                "sources": []}
    seen, sources = set(), []
    for c, _ in hits:
        if c["url"] not in seen and len(sources) < 4:
            seen.add(c["url"])
            sources.append({"title": c["title"] or c["url"], "url": c["url"]})
        if text.strip().lower().startswith("i don't have that information"):
            sources = []
    
    return {"answer": text, "sources": sources}
