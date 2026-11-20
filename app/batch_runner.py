import gc
from typing import Any, Callable, Dict, List, Tuple

from app.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP
from app.nlp_processor import chunk_text, lemmatize_text, normalize_text


def build_batch_runner(
    get_batch_targets_fn: Callable[[int, int], List[Tuple[str, str]]],
    scrape_page_fn: Callable[[str, str], str],
    upsert_chunks_fn: Callable[[List[Dict[str, Any]]], int],
    get_chunk_counts_fn: Callable[[], Tuple[int, Dict[str, int]]],
) -> Callable[[int, int], Dict[str, Any]]:
    def run_batch(batch_index: int, batch_size: int) -> Dict[str, Any]:
        targets = get_batch_targets_fn(batch_index, batch_size)
        prepared_chunks: List[Dict[str, Any]] = []
        scraped_pages = 0

        for site, url in targets:
            raw_content = scrape_page_fn(site, url)
            if not raw_content:
                continue
            scraped_pages += 1
            cleaned = normalize_text(raw_content)
            for idx, part in enumerate(chunk_text(cleaned, CHUNK_MAX_CHARS, CHUNK_OVERLAP)):
                prepared_chunks.append(
                    {
                        "id": f"{site}:{url}:{idx}",
                        "site": site,
                        "url": url,
                        "chunk_index": idx,
                        "text": part,
                        "lemma_text": lemmatize_text(part),
                    }
                )

        upserted = upsert_chunks_fn(prepared_chunks) if prepared_chunks else 0
        total_chunks, chunks_by_site = get_chunk_counts_fn()

        prepared_chunks.clear()
        del prepared_chunks
        gc.collect()

        return {
            "batch_index": batch_index,
            "pages": scraped_pages,
            "chunks": upserted,
            "total_chunks": total_chunks,
            "chunks_by_site": chunks_by_site,
            "changed": upserted > 0,
        }

    return run_batch