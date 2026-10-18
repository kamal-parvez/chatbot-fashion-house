"""Arize Phoenix / OpenInference tracing setup."""

from opentelemetry import trace as otel_trace
from openinference.instrumentation import OITracer, TraceConfig

from app import config

_tracer = None


def get_tracer() -> OITracer:
    """Returns the shared tracer, real if PHOENIX_ENABLED else a no-op."""
    global _tracer
    if _tracer is not None:
        return _tracer

    if config.PHOENIX_ENABLED:
        from phoenix.otel import register

        tracer_provider = register(
            project_name=config.PHOENIX_PROJECT_NAME,
            endpoint=config.PHOENIX_COLLECTOR_ENDPOINT,
            batch=True,  # async export so a missing collector can't add latency
            auto_instrument=False,
            set_global_tracer_provider=False,
            verbose=False,
        )
        _tracer = tracer_provider.get_tracer(__name__)
    else:
        _tracer = OITracer(wrapped=otel_trace.get_tracer(__name__), config=TraceConfig())

    return _tracer
