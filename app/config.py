"""App configuration, loaded from environment variables / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data"

FAQ_JOBLIB = DATASET_DIR / "faq.joblib"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_ROUTER_MODEL = os.environ.get("GEMINI_ROUTER_MODEL", "gemini-3.1-flash-lite")
GEMINI_ANSWER_MODEL = os.environ.get("GEMINI_ANSWER_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

WEAVIATE_DATA_PATH = os.environ.get("WEAVIATE_DATA_PATH", str(BASE_DIR / "weaviate_data"))

# Optional local folder of "{product_id}.jpg" images; shown in the UI if present.
PRODUCT_IMAGES_DIR = Path(os.environ.get("PRODUCT_IMAGES_DIR", str(DATASET_DIR / "images")))

# Caps product catalog size to keep first-run embedding fast on free-tier quota.
MAX_PRODUCTS = int(os.environ.get("MAX_PRODUCTS", "300"))

PRODUCTS_COLLECTION = "Products"
FAQ_COLLECTION = "Faq"

# Number of prior turns sent back to the LLM as conversation context.
CONTEXT_WINDOW = 20

# Traces export to a Phoenix instance run separately; see README.
PHOENIX_ENABLED = os.environ.get("PHOENIX_ENABLED", "true").lower() not in ("false", "0", "")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "fashion-chatbot")
PHOENIX_COLLECTOR_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
