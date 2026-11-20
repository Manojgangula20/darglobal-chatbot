import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE = Path(__file__).resolve().parent.parent
SEED_DIR = BASE / "seed"                      # committed snapshot: pages.json + emb_cache.pkl
DATA_DIR = Path(os.getenv("DATA_DIR", BASE / "runtime"))   # live data + embedding cache
MODEL_CACHE = BASE / "models"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_PRIMARY = os.getenv(
    "MODEL_PRIMARY",
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv("MODEL_FALLBACKS", "").split(",")
    if m.strip()
]

REFRESH_INTERVAL_MIN = int(os.getenv("REFRESH_INTERVAL_MIN", "360"))
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", ""))

# memory controls (the target host has about 512 MB RAM)
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "8"))          # chunks embedded per batch
MAX_PAGE_BYTES = int(os.getenv("MAX_PAGE_BYTES", "1500000"))   # skip pages larger than this

USER_AGENT = "DarGlobalAssignmentBot/1.0"
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))

_COMMON_EXCLUDE = [r"login", r"sign-?in", r"sign-?up", r"register", r"checkout", r"pay-online",
                   r"/cart", r"privacy", r"cookie", r"/search"]

SITES = {
    "darglobal": {
        "base": "https://darglobal.co.uk",
        "seeds": ["https://darglobal.co.uk/", "https://darglobal.co.uk/about",
                  "https://darglobal.co.uk/investor", "https://darglobal.co.uk/hospitality",
                  "https://darglobal.co.uk/projects"],
        "max_pages": 400,          # safety ceiling, not a target
        # English only: skip other language prefixes
        "exclude": [r"^/(ar|ru|fr|de|es|zh|it|tr)(/|$)"] + _COMMON_EXCLUDE,
    },
    "wasalt": {
        "base": "https://wasalt.sa",
        "seeds": ["https://wasalt.sa/en", "https://wasalt.sa/en/corporate"],
        "max_pages": 150,
        "include": [r"^/en(/|$)"],
        # skip thousands of changing property listings
        "exclude": [
            r"-for-",
            r"/propert",
            r"/ads?/",
            r"/listing",
            r"/unit",
            r"/map",
            r"/dailyrental",
        ] + _COMMON_EXCLUDE,
    },
}

BLOG = {
    "key": "wasalt-blog",
    "api": "https://blog.wasalt.sa/en/wp-json/wp/v2/posts",
    "max_posts": 600,              # the whole blog, with a safety ceiling
    "per_page": 50,
}
