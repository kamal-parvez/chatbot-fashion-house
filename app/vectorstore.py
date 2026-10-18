"""Weaviate embedded client: collection setup, insertion, and search."""

from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

from app import config

_client: Optional[weaviate.WeaviateClient] = None


def get_client() -> weaviate.WeaviateClient:
    """Returns the shared embedded Weaviate client, starting it if needed."""
    global _client
    if _client is None:
        _client = weaviate.connect_to_embedded(persistence_data_path=config.WEAVIATE_DATA_PATH)
    return _client


def close_client() -> None:
    """Closes the embedded Weaviate client, if open."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def ensure_collections(recreate: bool = False) -> None:
    """Creates the Products and Faq collections if they don't already exist."""
    client = get_client()

    if recreate:
        for name in (config.PRODUCTS_COLLECTION, config.FAQ_COLLECTION):
            if client.collections.exists(name):
                client.collections.delete(name)

    if not client.collections.exists(config.PRODUCTS_COLLECTION):
        client.collections.create(
            config.PRODUCTS_COLLECTION,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="product_id", data_type=DataType.INT),
                Property(name="productDisplayName", data_type=DataType.TEXT),
                Property(name="gender", data_type=DataType.TEXT),
                Property(name="masterCategory", data_type=DataType.TEXT),
                Property(name="subCategory", data_type=DataType.TEXT),
                Property(name="articleType", data_type=DataType.TEXT),
                Property(name="baseColour", data_type=DataType.TEXT),
                Property(name="season", data_type=DataType.TEXT),
                Property(name="usage", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.NUMBER),
                Property(name="price", data_type=DataType.NUMBER),
            ],
        )

    if not client.collections.exists(config.FAQ_COLLECTION):
        client.collections.create(
            config.FAQ_COLLECTION,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="question", data_type=DataType.TEXT),
                Property(name="answer", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
            ],
        )


def insert_products(products: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
    """Batch-inserts products with their embedding vectors."""
    collection = get_client().collections.get(config.PRODUCTS_COLLECTION)
    with collection.batch.dynamic() as batch:
        for product, vector in zip(products, vectors):
            batch.add_object(properties=product, vector=vector)


def insert_faq(faq_items: List[Dict[str, str]], vectors: List[List[float]]) -> None:
    """Batch-inserts FAQ entries with their embedding vectors."""
    collection = get_client().collections.get(config.FAQ_COLLECTION)
    with collection.batch.dynamic() as batch:
        for item, vector in zip(faq_items, vectors):
            batch.add_object(properties=item, vector=vector)


def count(collection_name: str) -> int:
    """Returns the number of objects stored in a collection."""
    return get_client().collections.get(collection_name).aggregate.over_all(total_count=True).total_count


def search_faq(query_vector: List[float], limit: int = 5):
    """Returns the nearest FAQ entries to a query vector."""
    collection = get_client().collections.get(config.FAQ_COLLECTION)
    return collection.query.near_vector(near_vector=query_vector, limit=limit).objects


def search_products(query_vector: List[float], filters=None, limit: int = 20):
    """Returns the nearest products to a query vector, optionally filtered."""
    collection = get_client().collections.get(config.PRODUCTS_COLLECTION)
    kwargs = {"near_vector": query_vector, "limit": limit}
    if filters:
        kwargs["filters"] = Filter.all_of(filters)
    return collection.query.near_vector(**kwargs).objects


def get_filter_by_metadata(json_output: Optional[dict] = None):
    """Builds Weaviate Filter objects from an LLM-generated metadata dict."""
    if json_output is None:
        return None

    valid_keys = (
        "gender",
        "masterCategory",
        "articleType",
        "baseColour",
        "price",
        "usage",
        "season",
    )

    filters = []
    for key, value in json_output.items():
        if key not in valid_keys:
            continue

        if key == "price":
            if not isinstance(value, dict):
                continue
            min_price = value.get("min")
            max_price = value.get("max")
            if min_price is None or max_price is None:
                continue
            if min_price <= 0 or max_price == "inf":
                continue
            filters.append(Filter.by_property(key).greater_than(min_price))
            filters.append(Filter.by_property(key).less_than(max_price))
        else:
            filters.append(Filter.by_property(key).contains_any(value))

    return filters
