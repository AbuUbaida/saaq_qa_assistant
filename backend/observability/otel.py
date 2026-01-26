"""OpenTelemetry setup helpers."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from .settings import get_settings


def setup_otel(app=None) -> None:
    """Configure OpenTelemetry tracing and optionally instrument FastAPI."""
    settings = get_settings()
    if not settings.enabled:
        return

    resource = Resource.create(attributes={"service.name": settings.service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set as GLOBAL provider
    trace.set_tracer_provider(provider)

    # Instrument outgoing HTTP calls (HF, Firecrawl, etc.)
    RequestsInstrumentor().instrument()

    # Instrument FastAPI incoming requests if app is provided.
    if app is not None:
        FastAPIInstrumentor.instrument_app(app)