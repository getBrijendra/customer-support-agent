"""Guideline upload -> parse -> compile -> validate -> examples -> embed -> persist -> retrieval
eval -> approve (auto by default) -> activate (latest by default).

`IngestionService.run_ingestion` has no FastAPI dependency so it can move to a worker queue
(Arq/Celery) unchanged: it only needs a version id and the injected collaborators.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy.exc import IntegrityError

from app.ai.compiler import PlaybookCompiler, ValidationReport, validate_playbook
from app.ai.examples import ExampleGenerator
from app.ai.prompts import COMPILE_PROMPT_VERSION
from app.ai.schemas import CompiledPlaybook, ExampleSet
from app.core.config import Settings
from app.core.errors import (
    CompanyNotFound,
    DuplicateGuidelines,
    FileTooLarge,
    InvalidFile,
    NoActiveGuidelines,
    VersionNotFound,
)
from app.core.logging import get_logger
from app.db.models import (
    Company,
    CompanyRules,
    GuidelineSection,
    GuidelineVersion,
    Intent,
    IntentExample,
    ReferenceFact,
    ReviewStatus,
    RoutingEvalCase,
    VersionStatus,
    utcnow,
)
from app.documents.base import ParsedDocument
from app.documents.registry import extension_of, get_parser
from app.services.common import AppMetrics, CompanyContextCache, UnitOfWork, UowFactory
from app.services.eval_metrics import ApprovalDecision, decide_approval
from app.services.eval_service import EvalService
from app.services.review_service import activate_if_latest

logger = get_logger(__name__)

_ALLOWED_CONTROL = {"\t", "\n", "\r", "\x0c"}
N_NEGATIVES = 4


class CompileFailure(Exception):
    pass


def _has_bad_control_chars(text: str) -> bool:
    return any(
        (ord(ch) < 32 and ch not in _ALLOWED_CONTROL) or 0x7F <= ord(ch) <= 0x9F for ch in text
    )


def decode_text(content: bytes) -> str:
    """UTF-8 (BOM tolerant) first; latin-1 only if it yields clean text."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
        if _has_bad_control_chars(text):
            raise InvalidFile("File is not valid UTF-8 or Latin-1 text") from None
        return text
    if "\x00" in text:
        raise InvalidFile("File looks binary, not text")
    return text


def validate_upload(
    filename: str | None, content: bytes, max_bytes: int, formats: set[str] | None = None
) -> str:
    """Validate an uploaded guideline file and return its decoded text."""
    get_parser(filename, formats)  # UNSUPPORTED_FORMAT for anything we can't parse
    if len(content) > max_bytes:
        raise FileTooLarge(f"File exceeds the {max_bytes} byte limit")
    text = decode_text(content)
    if not text.strip():
        raise InvalidFile("File is empty")
    return text


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class CreatedVersion:
    version: GuidelineVersion
    company_slug: str
    info: str | None = None


@dataclass
class Playbook:
    version: GuidelineVersion
    intents: list[Intent]
    rules: CompanyRules | None
    facts: list[ReferenceFact]
    sections: list[GuidelineSection]


def version_summary(v: GuidelineVersion) -> dict[str, object]:
    return {"version_number": v.version_number, "id": str(v.id), "status": v.status}


class IngestionService:
    def __init__(
        self,
        uow_factory: UowFactory,
        settings: Settings,
        cache: CompanyContextCache,
        metrics: AppMetrics,
        *,
        compiler: PlaybookCompiler,
        examples: ExampleGenerator,
        embeddings_for: Callable[[str], Embeddings],
        evals: EvalService,
    ) -> None:
        self._uow = uow_factory
        self._settings = settings
        self._cache = cache
        self._metrics = metrics
        self._compiler = compiler
        self._examples = examples
        self._embeddings_for = embeddings_for
        self._evals = evals

    # ------------------------------------------------------------------ upload
    async def create_version(
        self,
        *,
        slug: str,
        filename: str | None,
        content: bytes,
        notes: str | None,
        uploaded_by: str | None,
        activate: bool = True,
    ) -> CreatedVersion:
        text = validate_upload(
            filename, content, self._settings.MAX_UPLOAD_BYTES, self._settings.supported_formats
        )
        digest = sha256_hex(content)
        last_exc: Exception | None = None
        for _attempt in range(2):
            async with self._uow() as uow:
                try:
                    return await self._allocate(
                        uow,
                        slug,
                        filename or "",
                        text,
                        content,
                        digest,
                        notes,
                        uploaded_by,
                        activate,
                    )
                except IntegrityError as exc:  # lost a version-number race; retry once
                    await uow.rollback()
                    last_exc = exc
        raise RuntimeError("Could not allocate a version number") from last_exc

    async def _allocate(
        self,
        uow: UnitOfWork,
        slug: str,
        filename: str,
        text: str,
        content: bytes,
        digest: str,
        notes: str | None,
        uploaded_by: str | None,
        activate: bool,
    ) -> CreatedVersion:
        # Row lock on the company serialises version allocation for that company.
        company: Company | None = await uow.companies.get_by_slug_for_update(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        existing: list[GuidelineVersion] = await uow.versions.list_for_company(company.id)
        active = next((v for v in existing if v.id == company.active_version_id), None)
        if active is not None and active.content_sha256 == digest:
            raise DuplicateGuidelines(
                f"Content is identical to the active version v{active.version_number}",
                details={"existing_version": version_summary(active)},
            )
        info = None
        same = next(
            (v for v in existing if v.content_sha256 == digest and v.status == VersionStatus.READY),
            None,
        )
        if same is not None:
            info = (
                f"Content is identical to v{same.version_number}; you could activate that "
                "version instead of re-uploading."
            )
        number: int = await uow.versions.next_version_number(company.id)
        version = GuidelineVersion(
            id=uuid.uuid4(),
            company_id=company.id,
            version_number=number,
            status=VersionStatus.PROCESSING,
            review_status=ReviewStatus.PENDING,
            activate_on_approval=activate,
            file_name=filename,
            source_format=extension_of(filename) or "txt",
            raw_text=text,
            content_sha256=digest,
            byte_size=len(content),
            embedding_model=self._settings.EMBEDDING_MODEL,
            embedding_dimensions=self._settings.EMBEDDING_DIMENSIONS,
            notes=notes,
            uploaded_by=uploaded_by,
            created_at=utcnow(),
        )
        await uow.versions.add(version)
        await uow.commit()
        logger.info(
            "ingestion.started",
            extra={"company": slug, "version": number, "bytes": len(content)},
        )
        return CreatedVersion(version=version, company_slug=slug, info=info)

    # ------------------------------------------------------------- pipeline
    async def run_ingestion(self, version_id: uuid.UUID) -> GuidelineVersion | None:
        started = time.perf_counter()
        async with self._uow() as uow:
            version: GuidelineVersion | None = await uow.versions.get(version_id)
            if version is None:
                logger.warning("ingestion.version_missing", extra={"version_id": str(version_id)})
                return None
            company: Company | None = await uow.companies.get_by_id(version.company_id)
            assert company is not None
            previous = []
            if company.active_version_id:
                previous = [
                    {"key": i.key, "name": i.name, "description": i.description}
                    for i in await uow.playbooks.list_intents(company.active_version_id)
                ]
        slug, number = company.slug, version.version_number
        try:
            doc = get_parser(version.file_name).parse(version.raw_text)
            if not doc.sections:
                raise CompileFailure("Guideline produced no sections")
            logger.info(
                "ingestion.parsed",
                extra={"company": slug, "version": number, "sections": len(doc.sections)},
            )
            t_compile = time.perf_counter()
            compiled = await self._compiler.compile(
                doc, company_name=company.name, previous_intents=previous
            )
            report = validate_playbook(compiled.playbook, doc)
            if not report.ok:
                raise CompileFailure("; ".join(report.errors))
            compile_s = time.perf_counter() - t_compile
            self._metrics.compile_seconds.observe(compile_s)
            logger.info(
                "ingestion.compiled",
                extra={
                    "company": slug,
                    "version": number,
                    "intents": len(report.playbook.intents),
                    "facts": len(report.playbook.facts),
                    "warnings": len(report.warnings),
                    "duration_ms": int(compile_s * 1000),
                },
            )
            examples = (
                await self._examples.generate(
                    report.playbook.intents,
                    company_name=company.name,
                    n_index=self._settings.INTENT_INDEX_EXAMPLES,
                    n_eval=self._settings.INTENT_EVAL_EXAMPLES,
                    n_negatives=N_NEGATIVES if self._settings.INTENT_EVAL_EXAMPLES else 0,
                )
            ).examples
            await self._persist(
                company, version, doc, report, examples, compiled.model, int(compile_s * 1000)
            )
        except Exception as exc:
            return await self._fail(version_id, slug, number, started, exc)

        # Retrieval eval (cheap, no LLM) gates auto-approval.
        top1: float | None = None
        run_id = None
        try:
            async with self._uow() as uow:
                version = await uow.versions.get(version_id)
            assert version is not None
            run = await self._evals.run(company, version, "retrieval")
            top1, run_id = run.metrics.get("top1_accuracy"), run.id
            decision = decide_approval(
                auto_approve=self._settings.AUTO_APPROVE_GUIDELINES,
                blocking_errors=[],
                top1=top1,
                threshold=self._settings.AUTO_APPROVE_MIN_ROUTING_TOP1,
            )
        except Exception as exc:  # the version is still usable; an admin can approve it
            logger.warning(
                "ingestion.eval_failed",
                extra={"company": slug, "version": number, "error": type(exc).__name__},
            )
            report.warnings.append(f"Routing eval failed: {type(exc).__name__}: {exc}")
            decision = ApprovalDecision(False, "routing eval failed; approve manually")
        activated = False
        async with self._uow() as uow:
            locked: Company | None = await uow.companies.lock(company.id)
            version = await uow.versions.get(version_id)
            assert locked is not None and version is not None
            version.routing_eval_run_id = run_id
            version.validation_warnings = report.warnings
            version.ingest_duration_ms = int((time.perf_counter() - started) * 1000)
            if decision.approved:
                version.review_status = ReviewStatus.APPROVED
                version.reviewed_by = "auto"
                version.reviewed_at = utcnow()
                version.review_note = decision.reason
                if version.activate_on_approval:
                    activated = await activate_if_latest(uow, locked, version)
            else:
                version.review_note = decision.reason
            await uow.commit()
        self._cache.invalidate(slug)
        self._metrics.ingestions.labels(status="ready").inc()
        logger.info(
            "ingestion.completed",
            extra={
                "company": slug,
                "version": number,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "approved": decision.approved,
                "activated": activated,
                "eval_top1": top1,
            },
        )
        if decision.approved:
            logger.info("guidelines.approved", extra={"company": slug, "version": number})
        if activated:
            logger.info("guidelines.activated", extra={"company": slug, "version": number})
        return version

    async def _persist(
        self,
        company: Company,
        version: GuidelineVersion,
        doc: ParsedDocument,
        report: ValidationReport,
        examples: ExampleSet,
        compiler_model: str,
        compile_ms: int,
    ) -> None:
        pb: CompiledPlaybook = report.playbook
        cid, vid = company.id, version.id
        sections = [
            GuidelineSection(
                id=uuid.uuid4(),
                company_id=cid,
                version_id=vid,
                ref=s.ref,
                ordinal=s.ordinal,
                heading=s.heading,
                path=s.path,
                text=s.text,
                char_start=s.char_start,
                char_end=s.char_end,
            )
            for s in doc.sections
        ]
        ref_to_id = {s.ref: str(s.id) for s in sections}
        intents = [
            Intent(
                id=uuid.uuid4(),
                company_id=cid,
                version_id=vid,
                key=it.key,
                name=it.name,
                description=it.description,
                conditions=it.conditions,
                steps=[st.model_dump() for st in it.steps],
                reply_template=it.reply_template,
                placeholders=it.placeholders,
                source_section_ids=[ref_to_id[r] for r in it.source_sections],
            )
            for it in pb.intents
        ]
        intent_by_key = {i.key: i for i in intents}
        example_rows: list[IntentExample] = []
        eval_cases: list[RoutingEvalCase] = []
        for ex in examples.intents:
            intent = intent_by_key.get(ex.intent_key)
            if intent is None:
                continue
            index_texts = list(dict.fromkeys([intent.name, *ex.index_examples]))
            for t in index_texts:
                example_rows.append(
                    IntentExample(
                        id=uuid.uuid4(),
                        company_id=cid,
                        version_id=vid,
                        intent_id=intent.id,
                        text=t,
                        split="index",
                    )
                )
            for t in ex.eval_examples:
                example_rows.append(
                    IntentExample(
                        id=uuid.uuid4(),
                        company_id=cid,
                        version_id=vid,
                        intent_id=intent.id,
                        text=t,
                        split="eval",
                    )
                )
                eval_cases.append(_case(cid, vid, t, [intent.key]))
        # Intents without generated examples still get their name + description indexed.
        covered = {e.intent_id for e in example_rows}
        for intent in intents:
            if intent.id not in covered:
                for t in (intent.name, intent.description):
                    example_rows.append(
                        IntentExample(
                            id=uuid.uuid4(),
                            company_id=cid,
                            version_id=vid,
                            intent_id=intent.id,
                            text=t,
                            split="index",
                        )
                    )
        eval_cases.extend(_case(cid, vid, t, []) for t in examples.negatives)
        facts = [
            ReferenceFact(
                id=uuid.uuid4(),
                company_id=cid,
                version_id=vid,
                title=f.title,
                text=f.text,
                source_section_ids=[ref_to_id[r] for r in f.source_sections if r in ref_to_id],
            )
            for f in pb.facts
        ]
        rules = CompanyRules(
            id=uuid.uuid4(),
            company_id=cid,
            version_id=vid,
            rules=pb.rules.as_rules(),
            source_section_ids=[ref_to_id[r] for r in pb.rules.source_sections if r in ref_to_id],
        )

        # Embed everything for this version in one batch with the version's recorded model.
        to_embed: list[Any] = [*sections, *[e for e in example_rows if e.split == "index"], *facts]
        texts = [_embed_text(r) for r in to_embed]
        embeddings = self._embeddings_for(version.embedding_model)
        vectors = await asyncio.to_thread(embeddings.embed_documents, texts) if texts else []
        if len(vectors) != len(texts):
            raise CompileFailure("Embedding count mismatch")
        for row, vec in zip(to_embed, vectors, strict=True):
            if len(vec) != version.embedding_dimensions:
                raise CompileFailure(
                    f"Embedding has {len(vec)} dimensions, expected {version.embedding_dimensions}"
                )
            row.embedding = list(vec)

        async with self._uow() as uow:
            current = await uow.versions.get(vid)
            assert current is not None
            await uow.playbooks.add_all(sections)
            await uow.playbooks.add_all(intents)
            await uow.playbooks.add_all(example_rows)
            await uow.playbooks.add_all([rules, *facts])
            await uow.evals.add_cases(eval_cases)
            current.status = VersionStatus.READY
            current.ready_at = utcnow()
            current.section_count = len(sections)
            current.intent_count = len(intents)
            current.fact_count = len(facts)
            current.coverage = report.coverage
            current.validation_warnings = report.warnings
            current.compiler_model = compiler_model
            current.compile_prompt_version = COMPILE_PROMPT_VERSION
            current.compile_duration_ms = compile_ms
            await uow.commit()

    async def _fail(
        self, version_id: uuid.UUID, slug: str, number: int, started: float, exc: Exception
    ) -> GuidelineVersion | None:
        async with self._uow() as uow:
            version = await uow.versions.get(version_id)
            if version is not None:
                version.status = VersionStatus.FAILED
                version.error_message = f"{type(exc).__name__}: {exc}"[:2000]
                version.ingest_duration_ms = int((time.perf_counter() - started) * 1000)
                await uow.commit()
        self._metrics.ingestions.labels(status="failed").inc()
        logger.error(
            "ingestion.failed",
            extra={
                "company": slug,
                "version": number,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"[:500],
            },
        )
        return version

    # ----------------------------------------------------------------- reads
    async def list_versions(self, slug: str) -> tuple[Company, list[GuidelineVersion]]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            versions: list[GuidelineVersion] = await uow.versions.list_for_company(company.id)
            return company, versions

    async def get_version(self, slug: str, number: int) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            version = await uow.versions.get_by_number(company.id, number)
            if version is None:
                raise VersionNotFound(f"Version {number} not found for '{slug}'")
            return company, version

    async def get_active(self, slug: str) -> tuple[Company, GuidelineVersion]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            if company.active_version_id is None:
                raise NoActiveGuidelines(f"Company '{slug}' has no active guideline version")
            version = await uow.versions.get(company.active_version_id)
            assert version is not None
            return company, version

    async def get_playbook(self, slug: str, number: int) -> tuple[Company, Playbook]:
        async with self._uow() as uow:
            company = await self._company(uow, slug)
            version = await uow.versions.get_by_number(company.id, number)
            if version is None:
                raise VersionNotFound(f"Version {number} not found for '{slug}'")
            pb = Playbook(
                version=version,
                intents=await uow.playbooks.list_intents(version.id),
                rules=await uow.playbooks.get_rules(version.id),
                facts=await uow.playbooks.list_facts(version.id),
                sections=await uow.playbooks.list_sections(version.id),
            )
            return company, pb

    @staticmethod
    async def _company(uow: UnitOfWork, slug: str) -> Company:
        company: Company | None = await uow.companies.get_by_slug(slug)
        if company is None:
            raise CompanyNotFound(f"Company '{slug}' not found")
        return company


def _case(cid: uuid.UUID, vid: uuid.UUID, message: str, keys: list[str]) -> RoutingEvalCase:
    return RoutingEvalCase(
        id=uuid.uuid4(),
        company_id=cid,
        message=message,
        expected_intent_keys=keys,
        source="synthetic",
        created_from_version_id=vid,
        created_at=utcnow(),
    )


def _embed_text(row: Any) -> str:
    if isinstance(row, GuidelineSection):
        return row.text
    if isinstance(row, ReferenceFact):
        return f"{row.title}\n{row.text}"
    return str(row.text)
