import re
import unicodedata
from typing import List

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFKC", raw_text or "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def lemmatize_text(text: str) -> str:
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    lemmas: List[str] = []
    for tok in tokens:
        if len(tok) > 4 and tok.endswith("ies"):
            lemmas.append(tok[:-3] + "y")
        elif len(tok) > 5 and tok.endswith("ing"):
            lemmas.append(tok[:-3])
        elif len(tok) > 4 and tok.endswith("ed"):
            lemmas.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            lemmas.append(tok[:-1])
        else:
            lemmas.append(tok)
    return " ".join(lemmas)


def _find_split_point(text: str, start: int, target_end: int, length: int) -> int:
    if target_end >= length:
        return length
    min_cut = max(start + 1, target_end - min(120, target_end - start))
    split_pos = text.rfind(" ", min_cut, target_end + 1)
    if split_pos > start:
        return split_pos
    return target_end


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    safe_max = max(1, int(max_chars))
    safe_overlap = max(0, min(int(overlap), safe_max // 2))

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        raw_end = min(start + safe_max, length)
        end = _find_split_point(text, start, raw_end, length)
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= length:
            break
        next_start = max(end - safe_overlap, start + 1)
        while next_start < length and text[next_start] == " ":
            next_start += 1
        start = next_start

    return chunks