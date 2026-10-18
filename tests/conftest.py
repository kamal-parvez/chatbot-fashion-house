"""Shared pytest setup, applied before any test module imports `app.*`."""

import os

# Avoid background OTLP export noise during tests.
os.environ.setdefault("PHOENIX_ENABLED", "false")
