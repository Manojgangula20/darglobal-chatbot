"""Polite scraper: robots.txt respected, conditional GETs, size caps, change detection by content hash."""
import gc
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from . import config as C

log = logging.getLogger("scraper")
S = requests.Session()
S.headers.update({"User-Agent": C.USER_AGENT, "Accept-Language": "en"})
DROP = ["script", "style", "noscript", "svg", "nav", "header", "footer", "form", "iframe", "button", "aside"]
REQ_TIMEOUT = (10, 25)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- robots.txt ----------
_robots: dict[str, RobotFileParser] = {}


def _rp(root: str) -> RobotFileParser:
    if root not in _robots:
        rp = RobotFileParser()
        try:
            r = S.get(root + "/robots.txt", timeout=REQ_TIMEOUT)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except requests.RequestException:
            rp.parse([])
        _robots[root] = rp
    return _robots[root]


def robots_ok(url: str) -> bool:
    p = urlparse(url)
    return _rp(f"{p.scheme}://{p.netloc}").can_fetch(C.USER_AGENT, url)


# ---------- url helpers ----------
def norm(url: str) -> str:
    u, _ = urldefrag(url)
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/') or '/'}"


def same_host(a: str, b: str) -> bool:
    h = lambda x: urlparse(x).netloc.lower().removeprefix("www.")
    return h(a) == h(b)


def path_ok(cfg: dict, path: str) -> bool:
    if any(re.search(x, path, re.I) for x in cfg.get("exclude", [])):
        return False
    inc = cfg.get("include")
    return not inc or any(re.search(x, path, re.I) for x in inc)


def sitemap_urls(root: str, max_maps: int = 8, max_urls: int = 4000) -> list[str]:
    """Read sitemaps as capped text with a regex: no XML tree, so memory stays flat."""
    out, seen = [], set()
    todo = list(_rp(root).site_maps() or [root + "/sitemap.xml"])
    while todo and len(seen) < max_maps and len(out) < max_urls:
        sm = todo.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        try:
            with S.get(sm, timeout=REQ_TIMEOUT, stream=True) as r:
                if r.status_code != 200:
                    continue
                body = r.raw.read(5_000_000, decode_content=True)
        except Exception:
            continue
        for u in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body.decode("utf-8", "ignore")):
            u = u.strip()
            if u.lower().endswith(".xml"):
                if u not in seen:
                    todo.append(u)
            elif len(out) < max_urls:
                out.append(u)
    return out


# ---------- extraction ----------
def clean_lines(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= 3 and (not lines or lines[-1] != ln):
            lines.append(ln)
    return "\n".join(lines)


def _format_tables(root) -> None:
    """Convert HTML tables into readable 'Header: Value | ...' rows before flattening."""
    for tbl in root.find_all("table"):
        headers = [
            re.sub(r"\s+", " ", th.get_text(" ", strip=True))
            for th in tbl.find_all("th")
        ]
        formatted_rows = []
        for tr in tbl.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", td.get_text(" ", strip=True))
                for td in tr.find_all(["td", "th"])
            ]
            cells = [c for c in cells if c]
            if not cells:
                continue
            if headers and len(headers) == len(cells) and cells != headers:
                pairs = [f"{h}: {v}" for h, v in zip(headers, cells) if h != v]
                formatted_rows.append(" | ".join(pairs) if pairs else " | ".join(cells))
            else:
                formatted_rows.append(" | ".join(cells))
        if formatted_rows:
            tbl.replace_with("\n" + "\n".join(formatted_rows) + "\n")
        else:
            tbl.decompose()


def _extract_json_ld(soup: BeautifulSoup) -> str:
    """Pull readable text fields from JSON-LD blocks (often used on real estate & article pages)."""
    snippets = []
    for tag in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip() or len(raw) > 200_000:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                parts = []
                for k in ("@type", "name", "headline", "description", "address", "price", "priceCurrency"):
                    v = obj.get(k)
                    if isinstance(v, (str, int, float)) and str(v).strip():
                        parts.append(f"{k}: {str(v).strip()}")
                if parts:
                    snippets.append(" | ".join(parts[:6]))
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(obj, list):
                for item in obj[:25]:
                    walk(item)

        walk(data)
        if len(snippets) >= 20:
            break
    return "\n".join(snippets[:20])


def extract(html: str):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.sub(r"\s+", " ", title).strip()

    desc_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    description = re.sub(r"\s+", " ", desc_tag.get("content", "")).strip() if desc_tag else ""

    ld_text = _extract_json_ld(soup)
    links = [a["href"] for a in soup.find_all("a", href=True)]
    for t in soup(DROP):
        t.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    _format_tables(root)

    body = clean_lines(root.get_text("\n"))
    combined = "\n".join(x for x in (description, ld_text, body) if x)
    text = clean_lines(combined)
    soup.decompose()
    return title, text, links


def _get(url: str, rec: dict | None):
    """Streamed GET with a size cap. Returns (response, body); body is None when the page is too large."""
    headers = {}
    if rec and rec.get("etag"):
        headers["If-None-Match"] = rec["etag"]
    if rec and rec.get("last_modified"):
        headers["If-Modified-Since"] = rec["last_modified"]
    for attempt in range(2):
        try:
            r = S.get(url, headers=headers, timeout=REQ_TIMEOUT, stream=True)
            if r.status_code >= 500 and attempt == 0:
                r.close()
                time.sleep(2)
                continue
            body = b""
            if r.status_code == 200:
                parts, size = [], 0
                for part in r.iter_content(65536):
                    size += len(part)
                    if size > C.MAX_PAGE_BYTES:
                        r.close()
                        return r, None
                    parts.append(part)
                body = b"".join(parts)
            r.close()
            return r, body
        except requests.RequestException:
            if attempt:
                raise
            time.sleep(2)


# ---------- site scrape ----------
def scrape_site(key: str, existing: dict, limit: int | None = None):
    cfg = C.SITES[key]
    base, maxp = cfg["base"], limit or cfg["max_pages"]
    old = {u: r for u, r in existing.items() if r["site"] == key}
    queue, queued = [], set()

    def enq(u):
        u = norm(urljoin(base + "/", u))
        if u not in queued and same_host(u, base) and path_ok(cfg, urlparse(u).path):
            queued.add(u)
            queue.append(u)

    for s in cfg["seeds"]:
        enq(s)
    for u in sorted(sitemap_urls(base), key=lambda x: x.count("/")):
        enq(u)

    pages = {}
    rep = {"fetched": 0, "unchanged": 0, "skipped_robots": [], "thin": [], "errors": []}
    i = 0
    max_requests = maxp * 4
    while i < len(queue) and len(pages) < maxp and i < max_requests:
        url = queue[i]
        i += 1
        if i % 10 == 1:
            print(f"[{key}] request {i}/{len(queue)} | kept {len(pages)}/{maxp} | {url}", flush=True)
        if not robots_ok(url):
            rep["skipped_robots"].append(url)
            continue
        rec = old.get(url)
        try:
            r, body = _get(url, rec)
        except requests.RequestException as e:
            rep["errors"].append(f"{url}: {e}")
            if rec:
                pages[url] = rec
            continue
        time.sleep(C.REQUEST_DELAY)

        if r.status_code == 304 and rec:
            rec["fetched_at"] = now_iso()
            pages[url] = rec
            rep["unchanged"] += 1
            for l in rec.get("links", []):
                enq(l)
            continue
        if body is None:
            rep["errors"].append(f"{url}: skipped, over size limit")
            if rec:
                pages[url] = rec
            continue
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "") or not same_host(r.url, base):
            rep["errors"].append(f"{url}: HTTP {r.status_code}")
            if rec:
                pages[url] = rec
            continue

        title, text, links = extract(body.decode(r.encoding or "utf-8", "ignore"))
        del body
        abs_links = [norm(urljoin(url, l)) for l in links if not l.startswith(("mailto:", "tel:", "javascript:"))]
        abs_links = [l for l in abs_links if same_host(l, base) and path_ok(cfg, urlparse(l).path)][:120]
        for l in abs_links:
            enq(l)
        if len(text) < 80:
            rep["thin"].append(url)
            continue

        pages[url] = {
            "url": url,
            "site": key,
            "title": title,
            "text": text,
            "hash": sha(title + text),
            "etag": r.headers.get("ETag"),
            "last_modified": r.headers.get("Last-Modified"),
            "fetched_at": now_iso(),
            "links": abs_links,
        }
        rep["fetched"] += 1
    return pages, rep


# ---------- Wasalt blog (WordPress REST API), posts + pages ----------
def _wp_endpoint_items(endpoint: str, max_items: int, per_page: int, site_key: str, rep: dict) -> dict:
    items = {}
    page = 1
    while len(items) < max_items and page <= 40:
        r = None
        for attempt in range(2):
            try:
                r = S.get(
                    endpoint,
                    timeout=REQ_TIMEOUT,
                    params={
                        "per_page": per_page,
                        "page": page,
                        "orderby": "modified",
                        "order": "desc",
                        "_fields": "id,link,modified_gmt,title,content",
                    },
                )
                if r.status_code >= 500 and attempt == 0:
                    r.close()
                    time.sleep(2)
                    continue
                break
            except requests.RequestException as e:
                if attempt == 0:
                    time.sleep(2)
                    continue
                rep["errors"].append(f"{endpoint} page {page}: {e}")
                r = None
                break

        if r is None or r.status_code != 200:
            if r is not None and page == 1 and endpoint.endswith("/posts"):
                rep["errors"].append(f"blog API HTTP {r.status_code}")
            break

        try:
            data = r.json()
        except Exception as e:
            rep["errors"].append(f"{endpoint} page {page} json error: {e}")
            break

        if not data:
            break

        for p in data:
            title_soup = BeautifulSoup(p.get("title", {}).get("rendered", ""), "lxml")
            title = re.sub(r"\s+", " ", title_soup.get_text(" ", strip=True)).strip()
            title_soup.decompose()

            body_soup = BeautifulSoup(p.get("content", {}).get("rendered", ""), "lxml")
            _format_tables(body_soup)
            text = clean_lines(body_soup.get_text("\n"))
            body_soup.decompose()

            if len(text) < 80:
                continue
            url = norm(p["link"])
            items[url] = {
                "url": url,
                "site": site_key,
                "title": title,
                "text": text,
                "hash": sha(title + text),
                "modified": p.get("modified_gmt"),
                "fetched_at": now_iso(),
                "links": [],
            }
            rep["fetched"] += 1
            if len(items) >= max_items:
                break

        del data
        page += 1
        time.sleep(C.REQUEST_DELAY)
    return items


def scrape_blog(limit: int | None = None):
    cfg = C.BLOG
    pages, rep = {}, {"fetched": 0, "errors": [], "skipped_robots": []}
    if not robots_ok(cfg["api"]):
        rep["skipped_robots"].append(cfg["api"])
        return pages, rep

    maxp = limit or cfg["max_posts"]
    per_page = cfg.get("per_page", 50)
    pages.update(_wp_endpoint_items(cfg["api"], maxp, per_page, cfg["key"], rep))

    remaining = maxp - len(pages)
    if remaining > 0 and cfg["api"].endswith("/posts"):
        pages_api = cfg["api"][:-6] + "/pages"
        pages.update(_wp_endpoint_items(pages_api, remaining, per_page, cfg["key"], rep))
    return pages, rep


# ---------- orchestration with safety checks ----------
def _accept(key, pages, existing, rep, result, limit):
    old = {u: r for u, r in existing.items() if r["site"] == key}
    size = lambda d: sum(len(r["text"]) for r in d.values())
    if old and not limit and (len(pages) < 0.5 * len(old) or size(pages) < 0.5 * size(old)):
        rep["rejected"] = "new scrape far smaller than last good data; kept old data"
        result.update(old)
    else:
        result.update(pages)
    rep["pages"] = sum(1 for r in result.values() if r["site"] == key)


def scrape_all(existing: dict, limit: int | None = None):
    result, report = {}, {}
    for key in C.SITES:
        try:
            pages, rep = scrape_site(key, existing, limit)
        except Exception as e:
            log.exception("scrape failed for %s", key)
            pages, rep = {}, {"errors": [f"crash: {e}"]}
            if not limit:
                pages = {u: r for u, r in existing.items() if r["site"] == key}
        _accept(key, pages, existing, rep, result, limit)
        report[key] = rep
        gc.collect()

    try:
        pages, rep = scrape_blog(limit)
    except Exception as e:
        log.exception("blog scrape failed")
        pages, rep = {}, {"errors": [f"crash: {e}"]}
    _accept(C.BLOG["key"], pages, existing, rep, result, limit)
    report[C.BLOG["key"]] = rep
    gc.collect()
    return result, report