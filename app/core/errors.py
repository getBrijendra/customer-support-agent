"""Domain exceptions and the uniform `{"error": {...}}` response format."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)


class DomainError(Exception):
    code = "INTERNAL_ERROR"
    status_code = 500
    default_message = "Internal error"

    def __init__(self, message: str | None = None, *, details: Any = None) -> None:
        self.message = message or self.default_message
        self.details = details
        super().__init__(self.message)


class CompanyNotFound(DomainError):
    code, status_code, default_message = "COMPANY_NOT_FOUND", 404, "Company not found"


class CompanyAlreadyExists(DomainError):
    code, status_code, default_message = "COMPANY_EXISTS", 409, "Company slug already exists"


class VersionNotFound(DomainError):
    code, status_code, default_message = "VERSION_NOT_FOUND", 404, "Guideline version not found"


class ChatNotFound(DomainError):
    code, status_code, default_message = "CHAT_NOT_FOUND", 404, "Chat not found"


class NoActiveGuidelines(DomainError):
    code, status_code = "NO_ACTIVE_GUIDELINES", 409
    default_message = "Company has no active, ready guideline version"


class VersionNotReady(DomainError):
    code, status_code, default_message = "VERSION_NOT_READY", 409, "Version is not ready"


class VersionIsActive(DomainError):
    code, status_code, default_message = "VERSION_IS_ACTIVE", 409, "Cannot delete active version"


class DuplicateGuidelines(DomainError):
    code, status_code = "DUPLICATE_GUIDELINES", 409
    default_message = "Uploaded content is identical to the active guideline version"


class InvalidFile(DomainError):
    code, status_code, default_message = "INVALID_FILE", 422, "Invalid guideline file"


class FileTooLarge(DomainError):
    code, status_code, default_message = "FILE_TOO_LARGE", 413, "Guideline file is too large"


class Unauthorized(DomainError):
    code, status_code, default_message = "UNAUTHORIZED", 401, "Missing or invalid API key"


class Forbidden(DomainError):
    code, status_code, default_message = "FORBIDDEN", 403, "API key lacks the required role"


class LLMError(DomainError):
    code, status_code, default_message = "LLM_ERROR", 502, "The AI provider failed"


class RetrievalError(DomainError):
    code, status_code, default_message = "RETRIEVAL_ERROR", 502, "Guideline retrieval failed"


class UnsupportedFormat(DomainError):
    code, status_code, default_message = "UNSUPPORTED_FORMAT", 422, "Unsupported file format"


class VersionNotApproved(DomainError):
    code, status_code, default_message = "VERSION_NOT_APPROVED", 409, "Version is not approved"


class CompileError(DomainError):
    code, status_code, default_message = "COMPILE_ERROR", 422, "Guideline compilation failed"


class CapabilityNotBound(DomainError):
    code, status_code = "CAPABILITY_NOT_BOUND", 409
    default_message = "Capability has no live binding for this company"


class UnknownCapability(DomainError):
    code, status_code, default_message = "VALIDATION_ERROR", 422, "Unknown capability"


class BindingNotTested(DomainError):
    code, status_code = "BINDING_NOT_TESTED", 409
    default_message = "Binding must pass a test before going live"


class PolicyViolation(DomainError):
    code, status_code, default_message = "POLICY_VIOLATION", 409, "Action violates policy"


class ActionNotPending(DomainError):
    code, status_code, default_message = "ACTION_NOT_PENDING", 409, "Action is not pending"


class ToolError(DomainError):
    code, status_code, default_message = "TOOL_ERROR", 502, "Capability call failed"


class StorageError(DomainError):
    code, status_code, default_message = "STORAGE_ERROR", 502, "File storage failed"


class FileNotStored(DomainError):
    code, status_code = "FILE_NOT_STORED", 404
    default_message = "The original file was not stored for this version"


class NotFound(DomainError):
    code, status_code, default_message = "NOT_FOUND", 404, "Not found"


class Conflict(DomainError):
    code, status_code, default_message = "CONFLICT", 409, "Conflict"


class SecretsUnavailable(DomainError):
    code, status_code = "VALIDATION_ERROR", 422
    default_message = "SECRETS_ENCRYPTION_KEY is not configured"


class ValidationFailed(DomainError):
    code, status_code, default_message = "VALIDATION_ERROR", 422, "Invalid request"


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details is not None:
        body["details"] = details
    return {"error": body}


_HTTP_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    # 409: "CONFLICT",
    413: "FILE_TOO_LARGE",
    # 422: "VALIDATION_ERROR",
    # 500: "INTERNAL_ERROR",
    # 502: "BAD_GATEWAY",
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.code, exc.message, exc.details), status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return JSONResponse(
            error_body("VALIDATION_ERROR", "Request validation failed", details), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            error_body(code, str(exc.detail)), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"error_type": type(exc).__name__})
        return JSONResponse(error_body("INTERNAL_ERROR", "Internal server error"), status_code=500)
