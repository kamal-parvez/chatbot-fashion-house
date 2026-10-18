"""Thin wrapper around the Gemini API, returning (text, total_tokens)."""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types
from opentelemetry.trace import Status, StatusCode

from app import config
from app.tracing import get_tracer

_client = None
tracer = get_tracer()

# Gemini uses "model" for what we call "assistant".
_ROLE_MAP = {"assistant": "model", "user": "user", "system": "user"}

_MAX_RETRIES = 5
_DEFAULT_RETRY_SECONDS = 15
_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _get_client() -> genai.Client:
    """Returns the process-wide Gemini client, constructing it on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _generate_with_retry(client: genai.Client, **kwargs: Any):
    """Retries on transient errors: 429 rate limits and 503 model-overloaded."""
    for attempt in range(_MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except (errors.ClientError, errors.ServerError) as e:
            transient = "RESOURCE_EXHAUSTED" in str(e) or "UNAVAILABLE" in str(e)
            if not transient or attempt == _MAX_RETRIES - 1:
                raise
            match = _RETRY_DELAY_RE.search(str(e))
            wait_seconds = float(match.group(1)) + 2 if match else _DEFAULT_RETRY_SECONDS * (attempt + 1)
            print(f"  LLM call failed transiently, retrying in {wait_seconds:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})...")
            time.sleep(wait_seconds)


def _build_config(
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: int,
    system_instruction: Optional[str],
) -> types.GenerateContentConfig:
    """Builds a GenerateContentConfig, omitting unset sampling params."""
    kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    return types.GenerateContentConfig(**kwargs)


def _traced_call(span_name: str, model: str, contents: Any, gen_config: types.GenerateContentConfig) -> Tuple[str, int]:
    """Shared call path for `generate`/`chat_generate`, wrapped in a trace span."""
    client = _get_client()
    with tracer.start_as_current_span(span_name, openinference_span_kind="llm") as span:
        span.set_attribute("llm.model_name", model)
        span.set_attribute("llm.provider", "google")
        span.set_input(str(contents))
        try:
            response = _generate_with_retry(client, model=model, contents=contents, config=gen_config)
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))
            raise

        usage = response.usage_metadata
        total_tokens = usage.total_token_count if usage else 0
        text = response.text or ""

        if usage:
            span.set_attribute("llm.token_count.prompt", usage.prompt_token_count or 0)
            span.set_attribute("llm.token_count.completion", usage.candidates_token_count or 0)
            span.set_attribute("llm.token_count.total", total_tokens)
        span.set_output(text)
        span.set_status(Status(StatusCode.OK))

    return text, total_tokens


def generate(
    prompt: str,
    role: str = "user",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: int = 500,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
    **_ignored: Any,
) -> Tuple[str, int]:
    """Single-turn generation. Returns (text, total_tokens)."""
    return _traced_call(
        "llm_call",
        model or config.GEMINI_ANSWER_MODEL,
        prompt,
        _build_config(temperature, top_p, max_tokens, system_instruction),
    )


def chat_generate(
    history: List[Dict[str, str]],
    prompt: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: int = 500,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
    **_ignored: Any,
) -> Tuple[str, int]:
    """Multi-turn generation: `prompt` is appended to `history` as the newest turn."""
    contents = [
        types.Content(role=_ROLE_MAP.get(m["role"], "user"), parts=[types.Part(text=m["content"])])
        for m in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    return _traced_call(
        "llm_call",
        model or config.GEMINI_ANSWER_MODEL,
        contents,
        _build_config(temperature, top_p, max_tokens, system_instruction),
    )
