"""OpenTelemetry settings (simple, env-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


@dataclass(frozen=True)
class OTelSettings:
    enabled: bool = _env_bool("OTEL_ENABLED", False)
    service_name: str = os.getenv("OTEL_SERVICE_NAME", "saaq-qa-assistant-backend")
    otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    otlp_protocol: str = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    sample_rate: float = _env_float("OTEL_SAMPLE_RATE", 1.0)
    log_level: str = os.getenv("OTEL_LOG_LEVEL", "INFO")


def get_settings() -> OTelSettings:
    return OTelSettings()
    