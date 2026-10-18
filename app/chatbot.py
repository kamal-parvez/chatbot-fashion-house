"""Conversation wrapper around the RAG pipeline."""

from typing import Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from app import config, llm, pipeline
from app.tracing import get_tracer

tracer = get_tracer()

SYSTEM_PROMPT = (
    "You are a friendly assistant from Fashion Forward Hub. It is a cloth store "
    "selling a variety of items. Your job is to answer questions related to FAQ or Products."
)
INITIAL_MESSAGE = "Hi! How can I help you?"


def _extract_cited_products(answer: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Picks the candidate products whose names appear in `answer`."""
    answer_lower = answer.lower()
    return [p for p in candidates if p["productDisplayName"].lower() in answer_lower]


class ChatBot:
    """Stateful, single-conversation chat session."""

    def __init__(self, model: str = None, context_window: int = config.CONTEXT_WINDOW):
        self.model = model or config.GEMINI_ANSWER_MODEL
        self.context_window = context_window
        self.history: List[Dict[str, Any]] = [{"role": "assistant", "content": INITIAL_MESSAGE, "products": []}]
        self.total_tokens = 0

    def chat(self, user_message: str) -> str:
        """Answers one user turn and appends both sides to `history`."""
        with tracer.start_as_current_span("agent_call", openinference_span_kind="agent") as span:
            span.set_input({"prompt": user_message})

            recent_history = self.history[-self.context_window :]

            kwargs, routing_tokens = pipeline.answer_query(user_message)

            answer, generation_tokens = llm.chat_generate(
                recent_history,
                kwargs["prompt"],  # engineered prompt, not the raw message -- keeps history clean
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                max_tokens=700,
                model=self.model,
                system_instruction=SYSTEM_PROMPT,
            )

            cited_products = _extract_cited_products(answer, kwargs.get("candidates", []))

            self.total_tokens += routing_tokens + generation_tokens
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": answer, "products": cited_products})

            span.set_output({"content": answer, "total_tokens": routing_tokens + generation_tokens})
            span.set_status(Status(StatusCode.OK))

        return answer

    def clear(self) -> None:
        """Resets the conversation back to its initial greeting."""
        self.history = [{"role": "assistant", "content": INITIAL_MESSAGE, "products": []}]
        self.total_tokens = 0
