"""Loads the product catalog and FAQ corpus from data/."""

import random
from typing import Any, Dict, List

import joblib

from app import config

_SAMPLE_SEED = 42


def load_products() -> List[Dict[str, Any]]:
    """Returns product records from the catalog, capped at MAX_PRODUCTS.

    Samples randomly (fixed seed) rather than taking a prefix, since the
    catalog is grouped by category and a prefix slice would badly skew coverage.
    """
    products = joblib.load(config.DATASET_DIR / "clothes_json.joblib")
    if config.MAX_PRODUCTS and config.MAX_PRODUCTS > 0 and config.MAX_PRODUCTS < len(products):
        products = random.Random(_SAMPLE_SEED).sample(products, config.MAX_PRODUCTS)
    return products


def load_faq() -> List[Dict[str, str]]:
    """Returns FAQ records with keys: question, answer, type."""
    return joblib.load(config.FAQ_JOBLIB)


def product_to_text(product: Dict[str, Any]) -> str:
    """Flattens a product record into a single string for embedding."""
    return (
        f"{product['productDisplayName']}. "
        f"Category: {product['masterCategory']} / {product['subCategory']} / {product['articleType']}. "
        f"Color: {product['baseColour']}. Gender: {product['gender']}. "
        f"Usage: {product['usage']}. Season: {product['season']}. Price: ${product['price']}."
    )


def faq_to_text(faq_item: Dict[str, str]) -> str:
    """Flattens an FAQ record into a single string for embedding."""
    return f"{faq_item['question']} {faq_item['answer']}"
