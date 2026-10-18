"""Wiring tests for app.pipeline and app.chatbot, no network needed."""

from app import chatbot, pipeline, router


class FakeObj:
    """Stand-in for a Weaviate query result object."""

    def __init__(self, properties):
        self.properties = properties


def test_answer_query_routes_to_faq(monkeypatch):
    monkeypatch.setattr(router, "check_if_faq_or_product", lambda q: ("FAQ", 5))
    monkeypatch.setattr(
        pipeline.retrieval,
        "get_relevant_faq",
        lambda q, limit=5: [{"question": "Q?", "answer": "A.", "type": "general"}],
    )

    kwargs, total_tokens = pipeline.answer_query("What is your return policy?")

    assert "RELEVANT FAQ ITEMS" in kwargs["prompt"]
    assert "Q?" in kwargs["prompt"]
    assert total_tokens == 5
    assert kwargs["temperature"] is None


def test_answer_query_routes_to_products(monkeypatch):
    monkeypatch.setattr(router, "check_if_faq_or_product", lambda q: ("Product", 3))
    monkeypatch.setattr(router, "decide_task_nature", lambda q: ("technical", 2))

    product = {
        "product_id": 1,
        "productDisplayName": "Blue Tee",
        "masterCategory": "Apparel",
        "usage": "Casual",
        "gender": "Men",
        "articleType": "Tshirts",
        "subCategory": "Topwear",
        "baseColour": "Blue",
        "season": "Summer",
        "year": 2020,
    }
    monkeypatch.setattr(pipeline.retrieval, "get_relevant_products", lambda q: ([FakeObj(product)], 7))

    kwargs, total_tokens = pipeline.answer_query("Do you have blue t-shirts?")

    assert "Blue Tee" in kwargs["prompt"]
    assert "Product ID: 1" in kwargs["prompt"]
    assert total_tokens == 3 + 2 + 7
    assert kwargs["temperature"] == 0.3  # technical params
    assert kwargs["top_p"] == 0.7
    assert kwargs["candidates"] == [product]


def test_answer_query_fallback_for_undefined(monkeypatch):
    monkeypatch.setattr(router, "check_if_faq_or_product", lambda q: ("undefined", 1))

    kwargs, total_tokens = pipeline.answer_query("What's the weather like?")

    assert "What's the weather like?" in kwargs["prompt"]
    assert total_tokens == 1


def test_chatbot_history_stores_raw_message_not_engineered_prompt(monkeypatch):
    monkeypatch.setattr(
        chatbot.pipeline, "answer_query", lambda q: ({"prompt": "ENGINEERED PROMPT WITH CONTEXT", "temperature": None, "top_p": None}, 4)
    )
    monkeypatch.setattr(chatbot.llm, "chat_generate", lambda history, prompt, **kw: ("the answer", 9))

    bot = chatbot.ChatBot()
    answer = bot.chat("raw user question")

    assert answer == "the answer"
    assert bot.total_tokens == 13
    assert bot.history[-2] == {"role": "user", "content": "raw user question"}
    assert bot.history[-1] == {"role": "assistant", "content": "the answer", "products": []}


def test_chatbot_only_attaches_cited_products(monkeypatch):
    candidates = [
        {"product_id": 1, "productDisplayName": "Red Tee"},
        {"product_id": 2, "productDisplayName": "Blue Tee"},
    ]
    monkeypatch.setattr(
        chatbot.pipeline,
        "answer_query",
        lambda q: ({"prompt": "...", "temperature": None, "top_p": None, "candidates": candidates}, 0),
    )
    monkeypatch.setattr(
        chatbot.llm, "chat_generate", lambda history, prompt, **kw: ("Here is the Red Tee for you.", 0)
    )

    bot = chatbot.ChatBot()
    bot.chat("show me a red tee")

    assert bot.history[-1]["products"] == [candidates[0]]
