"""Live router-accuracy eval against labeled examples; skipped without an API key."""

import csv
import os

import pytest

from app import config, router

FIXTURE_PATH = config.BASE_DIR / "tests" / "fixtures" / "faq_or_products.csv"

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set; skipping live router eval"
)


def _load_fixture():
    with open(FIXTURE_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_router_accuracy():
    rows = _load_fixture()
    correct = 0
    for row in rows:
        label, _ = router.check_if_faq_or_product(row["query"])
        if label == row["label"]:
            correct += 1
    accuracy = correct / len(rows)
    print(f"Router accuracy: {accuracy:.0%} ({correct}/{len(rows)})")
    assert accuracy >= 0.8
