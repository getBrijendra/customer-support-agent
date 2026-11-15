"""U1 (txt parser) and U6 (parser registry)."""

from __future__ import annotations

import pytest

from app.core.errors import UnsupportedFormat
from app.documents.registry import get_parser
from app.documents.txt_parser import TxtParser, heading_level, normalize_text

DOC = """\
TONE
Greet the customer by first name.

1. Shipping Issues
Approach to solve:
- Verify order details and tracking.
- Apologize for the delay.

Sample reply:
Hello [Name], sorry for the delay.

2. Returns and Refunds
Approach to solve:
- Process the refund.
"""


def test_sections_headings_and_offsets() -> None:
    doc = TxtParser().parse(DOC)
    assert [s.ref for s in doc.sections] == ["S1", "S2", "S3"]
    assert [s.heading for s in doc.sections] == ["TONE", "Shipping Issues", "Returns and Refunds"]
    for s in doc.sections:  # offsets point into the normalised text
        assert doc.text[s.char_start : s.char_end].strip() == s.text
    # "Approach to solve:" / "Sample reply:" are sub-headings, not section boundaries
    assert "Sample reply:" in doc.sections[1].text
    assert doc.sections[1].body.startswith("Approach to solve:")


def test_normalisation_and_preamble() -> None:
    doc = TxtParser().parse("Intro line  \r\nmore\r\n\r\n# Billing\r\nPay now\r\n")
    assert doc.text == "Intro line\nmore\n\n# Billing\nPay now"
    assert doc.sections[0].heading is None and doc.sections[0].text == "Intro line\nmore"
    assert doc.sections[1].heading == "Billing"


def test_nested_paths() -> None:
    doc = TxtParser().parse("# Billing\nx\n## Refunds\ny\n# Shipping\nz\n")
    assert [s.path for s in doc.sections] == ["Billing", "Billing > Refunds", "Shipping"]


def test_heading_levels() -> None:
    assert heading_level("1. Connection Failures") == 1
    assert heading_level("1.2 Sub topic") == 2
    assert heading_level("REFUNDS:") == 1
    assert heading_level("Approach to solve:") == 9
    assert heading_level("This is a normal sentence.") is None
    assert normalize_text("a \r\nb\r") == "a\nb"


def test_document_without_headings_is_one_section() -> None:
    doc = TxtParser().parse("just some text\nwith lines")
    assert len(doc.sections) == 1 and doc.sections[0].heading is None


@pytest.mark.parametrize("name", ["guide.txt", "GUIDE.TXT"])
def test_registry_accepts_txt(name: str) -> None:
    assert get_parser(name).format == "txt"


@pytest.mark.parametrize("name", ["guide.pdf", "guide.docx", "guide", None, "a.txt.exe"])
def test_registry_rejects_other_formats(name: str | None) -> None:
    with pytest.raises(UnsupportedFormat) as exc:
        get_parser(name)
    assert exc.value.code == "UNSUPPORTED_FORMAT" and exc.value.status_code == 422


def test_registry_respects_enabled_formats() -> None:
    with pytest.raises(UnsupportedFormat):
        get_parser("guide.txt", enabled=set())
    with pytest.raises(UnsupportedFormat, match="planned"):
        get_parser("x.pdf")
