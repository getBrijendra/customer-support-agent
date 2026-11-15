from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_resources
from app.db.session import ping
from app.resources import Resources
from app.schemas.api import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Liveness probe")
async def health() -> HealthOut:
    return HealthOut(status="ok")


@router.get(
    "/ready",
    response_model=HealthOut,
    summary="Readiness probe (DB + pgvector search index)",
    responses={503: {"model": HealthOut}},
)
async def ready(resources: Resources = Depends(get_resources)) -> JSONResponse:
    checks = {
        "database": await ping(resources.engine),
        "search_index": await resources.search.healthcheck(),
    }
    ok = all(checks.values())
    body = HealthOut(status="ok" if ok else "degraded", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)
