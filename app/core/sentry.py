"""Optional Sentry error reporting + log capture.

Off unless `SENTRY_DSN` is set. When on, the standard-library `logging` records the app already
emits are forwarded to Sentry:

- records at `SENTRY_LOG_EVENT_LEVEL` (default ERROR) and above become Sentry **issues**;
- records at `SENTRY_LOG_BREADCRUMB_LEVEL` (default INFO) and above are attached as
  **breadcrumbs** to the next issue;
- with `SENTRY_ENABLE_LOGS=true`, records at `SENTRY_LOGS_LEVEL` and above are also sent to
  **Sentry Logs** (structured, searchable).

Privacy: the app never logs customer messages or tool payloads at INFO (only lengths + hashes),
and here PII is off, request bodies and local variables are not sent, and API-key headers are
scrubbed. Sentry is still a third-party service; review `before_send` before widening anything.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger, request_id_var

if TYPE_CHECKING:
    from app.core.config import Settings

logger = get_logger(__name__)

# Header / field names Sentry's default denylist misses (it has `x_api_key`, not `x-api-key`).
_EXTRA_DENYLIST = ["x-api-key", "x_api_key", "service_role_key", "dsn", "dsn_secret"]
# Loggers whose records are too noisy or could carry payloads.
_IGNORED_LOGGERS = ("uvicorn.access", "httpx", "httpcore", "google_genai", "mcp")


def _level(name: str) -> int:
    value = logging.getLevelName(name.upper())
    if not isinstance(value, int):
        raise ValueError(f"Unknown log level: {name!r}")
    return value


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
    """Tag the event with our request id and drop anything that can hold customer content."""
    request_id = request_id_var.get()
    if request_id:
        event.setdefault("tags", {})["request_id"] = request_id
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("data", None)  # multipart uploads / chat bodies
        request.pop("cookies", None)
        request.pop("query_string", None)  # dashboard takes `?key=`
    return event


def _before_send_log(log: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request_id_var.get()
    if request_id:
        log.setdefault("attributes", {})["request_id"] = request_id
    return log


def init_sentry(settings: Settings) -> bool:
    """Initialise the Sentry SDK once per process. Returns True when Sentry is enabled."""
    if not settings.SENTRY_DSN:
        return False

    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration, ignore_logger
    from sentry_sdk.integrations.starlette import StarletteIntegration
    from sentry_sdk.scrubber import DEFAULT_DENYLIST, EventScrubber

    if sentry_sdk.is_initialized():
        return True

    for name in _IGNORED_LOGGERS:
        ignore_logger(name)

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN.get_secret_value(),
        environment=settings.SENTRY_ENVIRONMENT or settings.APP_ENV,
        release=settings.SENTRY_RELEASE,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        event_scrubber=EventScrubber(denylist=DEFAULT_DENYLIST + _EXTRA_DENYLIST, recursive=True),
        before_send=_before_send,  # type: ignore[arg-type]
        enable_logs=settings.SENTRY_ENABLE_LOGS,
        before_send_log=_before_send_log,  # type: ignore[arg-type]
        integrations=[
            LoggingIntegration(
                level=_level(settings.SENTRY_LOG_BREADCRUMB_LEVEL),
                event_level=_level(settings.SENTRY_LOG_EVENT_LEVEL),
                sentry_logs_level=_level(settings.SENTRY_LOGS_LEVEL),
            ),
            # Domain errors (4xx) are handled responses, not incidents.
            StarletteIntegration(failed_request_status_codes={*range(500, 600)}),
            FastApiIntegration(failed_request_status_codes={*range(500, 600)}),
        ],
    )
    logger.info(
        "sentry.enabled",
        extra={
            "environment": settings.SENTRY_ENVIRONMENT or settings.APP_ENV,
            "logs_enabled": settings.SENTRY_ENABLE_LOGS,
            "traces_sample_rate": settings.SENTRY_TRACES_SAMPLE_RATE,
        },
    )
    return True
