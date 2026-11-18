"""Polite scraper: robots.txt respected, conditional GETs, change detection by content hash."""
import hashlib, logging, re, time
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


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- robots.txt ----------
_robots: dict[str, RobotFileParser] = {}


def _rp(root: str) -> RobotFileParser:
    if root not in _robots:
        rp = RobotFileParser()
        try:
            r = S.get(root + "/robots.txt", timeout=15)
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


def sitemap_urls(root: str, max_maps: int = 10) -> list[str]:
    out, seen = [], set()
    todo = list(_rp(root).site_maps() or [root + "/sitemap.xml"])
    while todo and len(seen) < max_maps:
        sm = todo.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = S.get(sm, timeout=20)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "xml")
        except Exception:
            continue
        for loc in soup.find_all("loc"):
            u = loc.get_text(strip=True)
            (todo if u.lower().endswith(".xml") else out).append(u)
    return out


# ---------- extraction ----------
def clean_lines(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= 3 and (not lines or lines[-1] != ln):
            lines.append(ln)
    return "\n".join(lines)


def extract(html: str):
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else ""

    desc_tag = soup.find(
        "meta",
        attrs={"name": re.compile("^description$", re.I)}
    )
    description = desc_tag.get("content", "").strip() if desc_tag else ""

    links = [a["href"] for a in soup.find_all("a", href=True)]

    for t in soup(DROP):
        t.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    body = clean_lines(root.get_text("\n"))

    text = clean_lines(
        f"{description}\n{body}" if description else body
    )

    return title, text, links


def _get(url: str, rec: dict | None):
    headers = {}
    if rec and rec.get("etag"):
        headers["If-None-Match"] = rec["etag"]
    if rec and rec.get("last_modified"):
        headers["If-Modified-Since"] = rec["last_modified"]
    for attempt in range(2):
        try:
            r = S.get(url, headers=headers, timeout=25)
            if r.status_code < 500:
                return r
        except requests.RequestException:
            if attempt:
                raise
        time.sleep(2)
    return r


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
    max_requests = maxp * 3      # stop after 3x the page budget
    while i < len(queue) and len(pages) < maxp and i < max_requests:
        url = queue[i]
        i += 1
        print(f"[{key}] request {i}/{len(queue)} | kept {len(pages)}/{maxp} | {url}", flush=True)
        if not robots_ok(url):
            rep["skipped_robots"].append(url)
            continue
        rec = old.get(url)
        try:
            r = _get(url, rec)
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
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "") or not same_host(r.url, base):
            rep["errors"].append(f"{url}: HTTP {r.status_code}")
            if rec:
                pages[url] = rec
            continue

        title, text, links = extract(r.text)
        abs_links = [norm(urljoin(url, l)) for l in links if not l.startswith(("mailto:", "tel:", "javascript:"))]
        abs_links = [l for l in abs_links if same_host(l, base) and path_ok(cfg, urlparse(l).path)][:200]
        for l in abs_links:
            enq(l)
        if len(text) < 150:
            rep["thin"].append(url)       # probably JS-rendered or empty
            continue
        pages[url] = {"url": url, "site": key, "title": title, "text": text, "hash": sha(title + text),
                      "etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"),
                      "fetched_at": now_iso(), "links": abs_links}
        rep["fetched"] += 1
    return pages, rep


# ---------- Wasalt blog (WordPress REST API) ----------
def scrape_blog(limit: int | None = None):
    cfg = C.BLOG
    pages, rep = {}, {"fetched": 0, "errors": [], "skipped_robots": []}
    if not robots_ok(cfg["api"]):
        rep["skipped_robots"].append(cfg["api"])
        return pages, rep
    maxp, page = limit or cfg["max_posts"], 1
    while len(pages) < maxp:
        try:
            r = S.get(cfg["api"], timeout=25, params={
                "per_page": 20, "page": page, "orderby": "modified", "order": "desc",
                "_fields": "id,link,modified_gmt,title,content"})
        except requests.RequestException as e:
            rep["errors"].append(str(e))
            break
        if r.status_code != 200:
            if page == 1:
                rep["errors"].append(f"blog API HTTP {r.status_code}")
            break
        data = r.json()
        if not data:
            break
        for p in data:
            title = BeautifulSoup(p["title"]["rendered"], "lxml").get_text(strip=True)
            text = clean_lines(BeautifulSoup(p["content"]["rendered"], "lxml").get_text("\n"))
            if len(text) < 150:
                continue
            url = norm(p["link"])
            pages[url] = {"url": url, "site": cfg["key"], "title": title, "text": text,
                          "hash": sha(title + text), "modified": p.get("modified_gmt"),
                          "fetched_at": now_iso(), "links": []}
            rep["fetched"] += 1
            if len(pages) >= maxp:
                break
        page += 1
        time.sleep(C.REQUEST_DELAY)
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
        except Exception as e:                     # never lose good data on a crash
            log.exception("scrape failed for %s", key)
            pages, rep = {}, {"errors": [f"crash: {e}"]}
            if not limit:
                pages = {u: r for u, r in existing.items() if r["site"] == key}
        _accept(key, pages, existing, rep, result, limit)
        report[key] = rep
    try:
        pages, rep = scrape_blog(limit)
    except Exception as e:
        log.exception("blog scrape failed")
        pages, rep = {}, {"errors": [f"crash: {e}"]}
    _accept(C.BLOG["key"], pages, existing, rep, result, limit)
    report[C.BLOG["key"]] = rep
    return result, report
