import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
SEED_DIR = BASE / "seed"
DATA_DIR = Path(os.getenv("DATA_DIR", BASE / "runtime"))
MODEL_CACHE = BASE / "models"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_PRIMARY = os.getenv(
    "MODEL_PRIMARY",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv("MODEL_FALLBACKS", "").split(",")
    if m.strip()
]

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "darglobal_wasalt")

REFRESH_INTERVAL_MIN = int(os.getenv("REFRESH_INTERVAL_MIN", "360"))
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "7"))

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "8"))
MAX_PAGE_BYTES = int(os.getenv("MAX_PAGE_BYTES", "1500000"))

USER_AGENT = "DarGlobalAssignmentBot/1.0"
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))

_COMMON_EXCLUDE = [
    r"login",
    r"sign-?in",
    r"sign-?up",
    r"register",
    r"checkout",
    r"pay-online",
    r"/cart",
    r"privacy",
    r"cookie",
    r"/search",
]

_LISTING_EXCLUDE = [
    r"-for-",
    r"/propert",
    r"/ads?/",
    r"/listing",
    r"/unit",
    r"/map",
    r"/dailyrental",
]

SITES = {
    "darglobal": {
        "base": "https://darglobal.co.uk",
        "seeds": [
            "https://darglobal.co.uk/",
            "https://darglobal.co.uk/about",
            "https://darglobal.co.uk/investor",
            "https://darglobal.co.uk/hospitality",
            "https://darglobal.co.uk/projects",
            "https://darglobal.co.uk/destinations",
            "https://darglobal.co.uk/partnerships",
            "https://darglobal.co.uk/news",
            "https://darglobal.co.uk/contact",
        ],
        "max_pages": 800,
        "exclude": [r"^/(ar|ru|fr|de|es|zh|it|tr)(/|$)"] + _LISTING_EXCLUDE + _COMMON_EXCLUDE,
    },
    "wasalt": {
        "base": "https://wasalt.sa",
        "seeds": [
            "https://wasalt.sa/en",
            "https://wasalt.sa/en/corporate",
            "https://wasalt.sa/en/about",
            "https://wasalt.sa/en/services",
            "https://wasalt.sa/en/projects",
            "https://wasalt.sa/en/faq",
            "https://wasalt.sa/en/contact",
        ],
        "max_pages": 250,
        "include": [r"^/en(/|$)"],
        "exclude": _LISTING_EXCLUDE + _COMMON_EXCLUDE,
    },
}

BLOG = {
    "key": "wasalt-blog",
    "api": "https://blog.wasalt.sa/en/wp-json/wp/v2/posts",
    "max_posts": 800,
    "per_page": 50,
}

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
NUM_BATCHES = int(os.getenv("NUM_BATCHES", "100"))
BATCH_GAP_MIN = int(os.getenv("BATCH_GAP_MIN", "60"))
BATCH_REFRESH_INTERVAL_MIN = int(os.getenv("BATCH_REFRESH_INTERVAL_MIN", "360"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))