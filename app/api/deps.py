from __future__ import annotations

from fastapi import Request

from app.core.logging import request_id_var
from app.resources import Resources
from app.services.chat_history_service import ChatHistoryService
from app.services.company_service import CompanyService
from app.services.guidance_service import GuidanceService
from app.services.ingestion_service import IngestionService
from app.services.metrics_service import MetricsService


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_request_id() -> str | None:
    return request_id_var.get()


def company_service(request: Request) -> CompanyService:
    return get_resources(request).company_service()


def ingestion_service(request: Request) -> IngestionService:
    return get_resources(request).ingestion_service()


def guidance_service(request: Request) -> GuidanceService:
    return get_resources(request).guidance_service()


def chat_history_service(request: Request) -> ChatHistoryService:
    return get_resources(request).chat_history_service()


def metrics_service(request: Request) -> MetricsService:
    return get_resources(request).metrics_service()
