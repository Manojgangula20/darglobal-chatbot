"""python -m app.cli dry-run | scrape | build | ask "question" """
import argparse, json, sys
from . import config as C, indexer, rag, scraper, store


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run", help="scrape a few pages per site, print what was found, save nothing")
    d.add_argument("--limit", type=int, default=8)
    sub.add_parser("scrape", help="full scrape + upload to Qdrant; writes seed/pages.json")
    sub.add_parser("build", help="build/upload the index to Qdrant from saved pages")
    a = sub.add_parser("ask", help="ask a question locally via Qdrant + OpenRouter (needs OPENROUTER_API_KEY)")
    a.add_argument("question")
    args = ap.parse_args()

    if args.cmd == "dry-run":
        pages, report = scraper.scrape_all({}, limit=args.limit)
        print(json.dumps(report, indent=2)[:6000])
        for r in pages.values():
            print(f"{r['site']:12} {len(r['text']):6} chars  {r['url']}")
    elif args.cmd == "scrape":
        store.ensure_runtime()
        old = store.load_pages(C.SEED_DIR / "pages.json") or store.load_pages()
        pages, report = scraper.scrape_all(old)
        store.save_pages(pages, C.SEED_DIR / "pages.json")
        store.save_pages(pages)
        print(json.dumps({k: {x: (len(v) if isinstance(v, list) else v) for x, v in r.items()}
                          for k, r in report.items()}, indent=2))
        print(f"saved {len(pages)} pages; embedding & uploading chunks to Qdrant...")
        idx = indexer.build(pages)
        print(f"index ready: {getattr(idx, '_count', 0)} chunks")
    elif args.cmd == "build":
        store.ensure_runtime()
        idx = indexer.build(store.load_pages())
        print(f"index ready: {getattr(idx, '_count', 0)} chunks")
    elif args.cmd == "ask":
        store.ensure_runtime()
        indexer.ensure_collection()
        total, _ = indexer.collection_counts()
        idx = indexer.QdrantIndex(initial_count=total)
        out = rag.answer(args.question, [], idx)
        print(out["answer"], "\n\nSources:", *[s["url"] for s in out["sources"]], sep="\n")


if __name__ == "__main__":
    sys.exit(main())