"""Ties routing + retrieval together into a per-turn prompt-building step."""

from typing import Any, Dict, Tuple

from app import retrieval, router
from app.tracing import get_tracer

tracer = get_tracer()


@tracer.tool
def query_on_faq(query: str) -> Tuple[str, int]:
    """Builds the final prompt for an FAQ-labeled query."""
    faq_items = retrieval.get_relevant_faq(query, limit=5)
    faq_layout = retrieval.generate_faq_layout(faq_items)

    prompt = (
        "You will be provided with a query for a clothing store regarding FAQ. "
        "It will be provided relevant FAQ from the clothing store. "
        "Answer the query based on the relevant FAQ provided. They are ordered in "
        "decreasing relevance, so the first is the most relevant FAQ and the last is "
        "the least relevant. Answer the instruction based on them. You might use more "
        "than one question and answer to make your answer. Only answer the question and "
        "do not mention that you have access to a FAQ.\n"
        "<FAQ>\n"
        f"RELEVANT FAQ ITEMS:\n{faq_layout}\n"
        "</FAQ>\n"
        f"Query: {query}"
    )
    return prompt, 0


@tracer.tool
def query_on_products(query: str) -> Tuple[str, float, float, int, list]:
    """Builds the final prompt for a Product-labeled query."""
    total_tokens = 0

    task_label, tokens = router.decide_task_nature(query)
    total_tokens += tokens
    params = router.get_params_for_task(task_label)

    relevant_products, tokens = retrieval.get_relevant_products(query)
    total_tokens += tokens

    context = retrieval.generate_items_context(relevant_products)

    prompt = (
        "Given the available set of clothing products given by: "
        f"CLOTHING PRODUCTS AVAILABLE:\n{context}\n"
        "Answer the question that follows.\n"
        "Never use more than 5 clothing products available below to compose your answer.\n"
        "Refer to each product by its exact name as given (not by ID). Do not mention product IDs.\n"
        "The other information might be provided but not necessarily all of them, pick only the relevant ones for the given query.\n"
        f"QUERY: {query}"
    )
    candidates = [obj.properties for obj in relevant_products]
    return prompt, params["temperature"], params["top_p"], total_tokens, candidates


@tracer.chain
def answer_query(query: str) -> Tuple[Dict[str, Any], int]:
    """Routes the query and builds the final prompt + sampling params."""
    total_tokens = 0

    label, tokens = router.check_if_faq_or_product(query)
    total_tokens += tokens

    if label not in ("FAQ", "Product"):
        return (
            {
                "prompt": (
                    "User provided a question that does not fit FAQ or Product-related "
                    f"categories. Answer it based on the context you already have. "
                    f"Query provided by the user: {query}"
                ),
                "temperature": None,
                "top_p": None,
                "candidates": [],
            },
            total_tokens,
        )

    if label == "FAQ":
        prompt, tokens = query_on_faq(query)
        total_tokens += tokens
        return {"prompt": prompt, "temperature": None, "top_p": None, "candidates": []}, total_tokens

    try:
        prompt, temperature, top_p, tokens, candidates = query_on_products(query)
        total_tokens += tokens
        return {"prompt": prompt, "temperature": temperature, "top_p": top_p, "candidates": candidates}, total_tokens
    except Exception:
        return (
            {
                "prompt": (
                    "User provided a question that broke the querying system. Instruct "
                    "them to rephrase it. Answer it based on the context you already "
                    f"have. Query provided by the user: {query}"
                ),
                "temperature": None,
                "top_p": None,
                "candidates": [],
            },
            total_tokens,
        )
