"""U2 — upload validation maps to the right error codes."""

from __future__ import annotations

import pytest

from app.core.errors import DomainError, FileTooLarge, InvalidFile
from app.services.ingestion_service import decode_text, validate_upload


def test_valid_utf8() -> None:
    assert validate_upload("guide.txt", "Héllo policy".encode(), 1000) == "Héllo policy"


def test_utf8_bom_stripped() -> None:
    assert validate_upload("g.TXT", b"\xef\xbb\xbfRefunds", 1000) == "Refunds"


def test_latin1_fallback_when_clean() -> None:
    assert decode_text("Café crème".encode("latin-1")) == "Café crème"


@pytest.mark.parametrize("name", ["guide.pdf", "guide.txt.exe", "guide", None, ""])
def test_wrong_extension(name: str | None) -> None:
    with pytest.raises(DomainError) as exc:
        validate_upload(name, b"hello", 1000)
    assert exc.value.code == "UNSUPPORTED_FORMAT"


@pytest.mark.parametrize("content", [b"", b"   \n\t  \r\n"])
def test_empty_file(content: bytes) -> None:
    with pytest.raises(InvalidFile) as exc:
        validate_upload("g.txt", content, 1000)
    assert exc.value.code == "INVALID_FILE"


def test_oversized_file() -> None:
    with pytest.raises(FileTooLarge) as exc:
        validate_upload("g.txt", b"a" * 1001, 1000)
    assert exc.value.code == "FILE_TOO_LARGE"
    assert exc.value.status_code == 413


@pytest.mark.parametrize(
    "content",
    [
        bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x81, 0x9F]),  # binary
        b"\xff\xfe\x00h\x00i\x00\x01",
        b"valid start \x00 binary",
    ],
)
def test_bad_encoding(content: bytes) -> None:
    with pytest.raises(InvalidFile) as exc:
        validate_upload("g.txt", content, 1000)
    assert exc.value.code == "INVALID_FILE"
