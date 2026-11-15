"""Extension -> parser. Formats listed in UNSUPPORTED are known but not implemented yet."""

from __future__ import annotations

from app.core.errors import UnsupportedFormat
from app.documents.base import DocumentParser
from app.documents.txt_parser import TxtParser

_PARSERS: dict[str, DocumentParser] = {"txt": TxtParser()}
UNSUPPORTED = {"pdf", "docx", "doc", "md", "html"}


def extension_of(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def get_parser(filename: str | None, enabled: set[str] | None = None) -> DocumentParser:
    ext = extension_of(filename)
    enabled = enabled if enabled is not None else set(_PARSERS)
    parser = _PARSERS.get(ext)
    if parser is None or ext not in enabled:
        supported = ", ".join(sorted(f".{e}" for e in enabled if e in _PARSERS))
        hint = " (planned, not implemented yet)" if ext in UNSUPPORTED else ""
        raise UnsupportedFormat(
            f"Unsupported guideline format '.{ext or '?'}'{hint}. Supported: {supported}"
        )
    return parser
