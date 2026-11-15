"""Admin: routing eval cases and runs."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Path, Query, status

from app.api.deps import eval_service
from app.core.security import Principal, require_admin
from app.schemas.api import (
    SLUG_PATTERN,
    ErrorResponse,
    EvalCaseIn,
    EvalCaseOut,
    EvalRunIn,
    EvalRunOut,
)
from app.services.eval_service import EvalService

router = APIRouter(tags=["evals"])
SlugPath = Path(pattern=SLUG_PATTERN, examples=["bookshop"])


@router.get("/companies/{slug}/eval-cases", response_model=list[EvalCaseOut])
async def list_eval_cases(
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: EvalService = Depends(eval_service),
) -> list[EvalCaseOut]:
    return [EvalCaseOut.build(c) for c in await service.list_cases(slug)]


@router.post(
    "/companies/{slug}/eval-cases",
    response_model=EvalCaseOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a manual routing eval case (empty keys = out of scope)",
)
async def add_eval_case(
    body: EvalCaseIn,
    slug: str = SlugPath,
    _: Principal = Depends(require_admin),
    service: EvalService = Depends(eval_service),
) -> EvalCaseOut:
    return EvalCaseOut.build(await service.add_case(slug, body.message, body.expected_intent_keys))


@router.post(
    "/internal/evals/routing",
    response_model=EvalRunOut,
    summary="Run routing evals for a version (default: active) — retrieval or full mode",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def run_routing_eval(
    body: EvalRunIn,
    _: Principal = Depends(require_admin),
    service: EvalService = Depends(eval_service),
) -> EvalRunOut:
    run, slug, number = await service.run_for_slug(body.company, body.version_number, body.mode)
    return EvalRunOut.build(run, slug, number)


@router.get("/internal/evals/routing/runs", response_model=list[EvalRunOut])
async def list_eval_runs(
    company: str | None = Query(None, pattern=SLUG_PATTERN),
    _: Principal = Depends(require_admin),
    service: EvalService = Depends(eval_service),
) -> list[EvalRunOut]:
    return [
        EvalRunOut.build(r, slug, n, include_results=False)
        for r, slug, n in await service.list_runs(company)
    ]


@router.get(
    "/internal/evals/routing/runs/{run_id}",
    response_model=EvalRunOut,
    responses={404: {"model": ErrorResponse}},
)
async def get_eval_run(
    run_id: uuid.UUID,
    _: Principal = Depends(require_admin),
    service: EvalService = Depends(eval_service),
) -> EvalRunOut:
    run, slug, number = await service.get_run(run_id)
    return EvalRunOut.build(run, slug, number)
