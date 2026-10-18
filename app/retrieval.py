"""FAQ and product retrieval: semantic search plus LLM-generated metadata filters."""

import json
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.trace import Status, StatusCode

from app import config, data, embeddings, llm, vectorstore
from app.tracing import get_tracer

tracer = get_tracer()

_possible_values: Optional[Dict[str, List[Any]]] = None


def _get_possible_values() -> Dict[str, List[Any]]:
    """Builds {field: [possible values]} from the indexed product catalog."""
    global _possible_values
    if _possible_values is None:
        products = data.load_products()
        values: Dict[str, set] = {}
        skip_keys = {"product_id", "price", "productDisplayName", "subCategory", "year"}
        for product in products:
            for key, val in product.items():
                if key in skip_keys:
                    continue
                values.setdefault(key, set()).add(val)
        _possible_values = {k: sorted(v) for k, v in values.items()}
    return _possible_values


def generate_faq_layout(faq_items: List[Dict[str, str]]) -> str:
    """Renders FAQ records as a plain-text block for inclusion in a prompt."""
    text = ""
    for item in faq_items:
        text += f"Question: {item['question']} Answer: {item['answer']} Type: {item['type']}\n"
    return text


def get_relevant_faq(query: str, limit: int = 5) -> List[Dict[str, str]]:
    with tracer.start_as_current_span("retrieve_faq_questions", openinference_span_kind="retriever") as span:
        span.set_input({"query": query, "limit": limit})

        query_vector = embeddings.embed_query(query)
        results = vectorstore.search_faq(query_vector, limit=limit)

        for i, obj in enumerate(results):
            span.set_attribute(f"retrieval.documents.{i}.document.id", str(obj.uuid))
            span.set_attribute(f"retrieval.documents.{i}.document.content", str(obj.properties))

        items = [obj.properties for obj in results]
        items.reverse()  # most relevant last, closer to the end of the prompt

        span.set_output({"count": len(items)})
        span.set_status(Status(StatusCode.OK))

    return items


def generate_items_context(results: List[Any]) -> str:
    """Renders retrieved product objects as a plain-text block for a prompt."""
    text = ""
    for obj in results:
        item = obj.properties
        text += (
            f"Product ID: {item['product_id']}. "
            f"Product name: {item['productDisplayName']}. "
            f"Product Category: {item['masterCategory']}. "
            f"Product usage: {item['usage']}. "
            f"Product gender: {item['gender']}. "
            f"Product Type: {item['articleType']}. "
            f"Product Category: {item['subCategory']} "
            f"Product Color: {item['baseColour']}. "
            f"Product Season: {item['season']}. "
            f"Product Year: {item['year']}.\n"
        )
    return text


def parse_json_output(llm_output: str) -> Optional[dict]:
    """Best-effort parse of an LLM's JSON-formatted reply."""
    try:
        cleaned = llm_output.replace("\n", "").replace("'", '"').replace("}}", "}").replace("{{", "{")
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def generate_metadata_from_query(query: str) -> Tuple[str, int]:
    """Asks the LLM to extract a structured product filter from a query."""
    values = _get_possible_values()
    prompt = f"""
One query will be provided. For the given query, there will be a call on vector database to query relevant cloth items.
Generate a JSON with useful metadata to filter the products in the query. Possible values for each feature is in the following json: {values}

Provide a JSON with the features that best fit in the query (can be more than one, write in a list). Also, if present, add a price key, saying if there is a price range (between values, greater than or smaller than some value).
Only return the JSON, nothing more. price key must be a json with "min" and "max" values (0 if no lower bound and inf if no upper bound).
Always include gender, masterCategory, ArticleType, baseColour, price, usage and season as keys. All values must be within lists.
If there is no price set, add min = 0 and max = inf.
Only include values that are given in the json above.

Example of expected JSON:

{{
"gender": ["Women"],
"masterCategory": ["Apparel"],
"articleType": ["Dresses"],
"baseColour": ["Blue"],
"price": {{"min": 0, "max": "inf"}},
"usage": ["Formal"],
"season": ["All seasons"]
}}

Query: {query}
         """
    with tracer.start_as_current_span("generate_metadata_from_query", openinference_span_kind="tool") as span:
        span.set_input({"query": query})
        text, total_tokens = llm.generate(prompt, temperature=0, max_tokens=1500, model=config.GEMINI_ROUTER_MODEL)
        span.set_output(text)
        span.set_status(Status(StatusCode.OK))
    return text, total_tokens


def generate_filters_from_query(query: str):
    """LLM-generates metadata and converts it into Weaviate Filter objects."""
    json_string, total_tokens = generate_metadata_from_query(query)
    json_output = parse_json_output(json_string)
    filters = vectorstore.get_filter_by_metadata(json_output)
    return filters, total_tokens


_IMPORTANCE_ORDER = ["baseColour", "masterCategory", "usage", "masterCategory", "season", "articleType", "gender"]


def _set_retrieval_attributes(span, results: List[Any]) -> None:
    """Records each retrieved document's id/content on a trace span."""
    for i, obj in enumerate(results):
        span.set_attribute(f"retrieval.documents.{i}.document.id", str(obj.uuid))
        span.set_attribute(f"retrieval.documents.{i}.document.content", str(obj.properties))


def get_relevant_products(query: str) -> Tuple[List[Any], int]:
    """Semantic search over products, filtered and relaxed if too few results."""
    with tracer.start_as_current_span("get_relevant_products_from_query", openinference_span_kind="retriever") as span:
        span.set_input({"query": query})

        query_vector = embeddings.embed_query(query)
        filters, total_tokens = generate_filters_from_query(query)

        if not filters:
            span.set_attribute("retrieval.filters", "")
            results = vectorstore.search_products(query_vector, limit=20)
            _set_retrieval_attributes(span, results)
            span.set_output({"count": len(results), "total_tokens": total_tokens})
            span.set_status(Status(StatusCode.OK))
            return results, total_tokens

        span.set_attribute("retrieval.filters", str(filters))
        results = vectorstore.search_products(query_vector, filters=filters, limit=20)
        _set_retrieval_attributes(span, results)

        if len(results) < 10:
            # Drop filters one at a time, least important first.
            for i in range(len(_IMPORTANCE_ORDER)):
                with tracer.start_as_current_span(f"refilter_{i}", openinference_span_kind="chain") as refilter_span:
                    remaining_targets = _IMPORTANCE_ORDER[i + 1 :]
                    filtered_filters = [f for f in filters if f.target in remaining_targets]
                    refilter_span.set_input(str(filtered_filters))

                    results = vectorstore.search_products(query_vector, filters=filtered_filters, limit=20)
                    _set_retrieval_attributes(refilter_span, results)

                    if len(results) >= 5:
                        refilter_span.set_output({"count": len(results)})
                        refilter_span.set_status(Status(StatusCode.OK))
                        span.set_output({"count": len(results), "total_tokens": total_tokens})
                        span.set_status(Status(StatusCode.OK))
                        return results, total_tokens

        span.set_output({"count": len(results), "total_tokens": total_tokens})
        span.set_status(Status(StatusCode.OK))
        return results, total_tokens
