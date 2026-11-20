import re
import unicodedata
from typing import List

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFKC", raw_text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    lemmas: List[str] = []
    for tok in tokens:
        if len(tok) > 4 and tok.endswith("ies"):
            lemmas.append(tok[:-3] + "y")
        elif len(tok) > 4 and tok.endswith("ing"):
            lemmas.append(tok[:-3])
        elif len(tok) > 3 and tok.endswith("ed"):
            lemmas.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            lemmas.append(tok[:-1])
        else:
            lemmas.append(tok)
    return " ".join(lemmas)


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks