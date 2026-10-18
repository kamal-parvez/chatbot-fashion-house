"""Text embeddings via the Gemini embeddings API."""

import re
import time
from typing import List

from google import genai
from google.genai import errors, types

from app import config

_client = None
_BATCH_SIZE = 50
_MAX_RETRIES = 6
_DEFAULT_RETRY_SECONDS = 30

_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _get_client() -> genai.Client:
    """Returns the process-wide Gemini client, constructing it on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _embed_batch_with_retry(client: genai.Client, batch: List[str], task_type: str):
    """Embeds one batch, retrying with the server's suggested backoff on 429s."""
    for attempt in range(_MAX_RETRIES):
        try:
            return client.models.embed_content(
                model=config.GEMINI_EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e) or attempt == _MAX_RETRIES - 1:
                raise
            match = _RETRY_DELAY_RE.search(str(e))
            wait_seconds = float(match.group(1)) + 2 if match else _DEFAULT_RETRY_SECONDS
            print(f"  Rate limited, retrying in {wait_seconds:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})...")
            time.sleep(wait_seconds)


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embeds a list of texts, batching requests to stay within API limits."""
    client = _get_client()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = _embed_batch_with_retry(client, batch, task_type)
        vectors.extend(e.values for e in response.embeddings)
    return vectors


def embed_query(text: str) -> List[float]:
    """Embeds a single search query (uses the query-optimized task type)."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
