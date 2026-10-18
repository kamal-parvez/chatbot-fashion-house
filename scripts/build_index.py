"""One-time script: embeds the catalog/FAQ and populates Weaviate."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, data, embeddings, vectorstore  # noqa: E402


def main(recreate: bool) -> None:
    """Indexes the FAQ corpus and product catalog into Weaviate."""
    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    vectorstore.ensure_collections(recreate=recreate)

    print(f"Loading FAQ corpus from {config.FAQ_JOBLIB} ...")
    faq_items = data.load_faq()
    print(f"  {len(faq_items)} FAQ entries")

    print(f"Loading product catalog (capped at MAX_PRODUCTS={config.MAX_PRODUCTS or 'unlimited'}) ...")
    products = data.load_products()
    print(f"  {len(products)} products")

    if vectorstore.count(config.FAQ_COLLECTION) == 0:
        print("Embedding and indexing FAQ ...")
        texts = [data.faq_to_text(f) for f in faq_items]
        vectors = embeddings.embed_texts(texts)
        vectorstore.insert_faq(faq_items, vectors)
    else:
        print("FAQ collection already populated, skipping (use --recreate to rebuild).")

    if vectorstore.count(config.PRODUCTS_COLLECTION) == 0:
        print("Embedding and indexing products (this calls the Gemini embeddings API in batches)...")
        start = time.time()
        texts = [data.product_to_text(p) for p in products]
        vectors = embeddings.embed_texts(texts)
        vectorstore.insert_products(products, vectors)
        print(f"  done in {time.time() - start:.1f}s")
    else:
        print("Products collection already populated, skipping (use --recreate to rebuild).")

    print(f"FAQ collection count:      {vectorstore.count(config.FAQ_COLLECTION)}")
    print(f"Products collection count: {vectorstore.count(config.PRODUCTS_COLLECTION)}")
    vectorstore.close_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild both collections")
    args = parser.parse_args()
    main(recreate=args.recreate)
