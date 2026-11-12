"""U1 — chunker: section metadata, deterministic ids, overlap."""

from __future__ import annotations

import itertools
import uuid

from app.ai.chunker import (
    chunk_guideline,
    find_headings,
    is_heading,
    make_chunk_id,
    normalize_text,
)

GUIDE = """\
TONE AND STYLE
Always greet the customer by first name. Sign off with "Warm regards, The Bookshop Team".
Be empathetic and concise.

REFUNDS:
Customers may request a refund within 30 days of delivery. Damaged books are refunded in full,
including shipping. Opened e-books are not refundable unless the file is corrupted.

# Shipping
Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.
International orders may take up to 14 days.

1. Escalation
Escalate to a team lead when a customer threatens legal action or asks for compensation over
50 EUR.
"""


def test_heading_detection() -> None:
    assert is_heading("TONE AND STYLE")
    assert is_heading("REFUNDS:")
    assert is_heading("# Shipping")
    assert is_heading("1. Escalation")
    assert is_heading("Returns policy:")
    assert not is_heading("Be empathetic and concise.")
    assert not is_heading("1. Always greet the customer by name.")
    assert not is_heading("")
    names = [h[1] for h in find_headings(GUIDE)]
    assert names == ["TONE AND STYLE", "REFUNDS", "Shipping", "1. Escalation"]


def test_section_metadata_attached() -> None:
    vid = uuid.uuid4()
    chunks = chunk_guideline(GUIDE, chunk_size=200, chunk_overlap=20, version_id=vid)
    assert len(chunks) >= 4
    by_text = {c.document.page_content: c.document.metadata["section"] for c in chunks}
    refund = next(s for t, s in by_text.items() if "30 days" in t)
    shipping = next(s for t, s in by_text.items() if "Express shipping" in t)
    escalation = next(s for t, s in by_text.items() if "legal action" in t)
    assert refund == "REFUNDS"
    assert shipping == "Shipping"
    assert escalation == "1. Escalation"


def test_chunk_ids_are_deterministic() -> None:
    vid = uuid.uuid4()
    a = chunk_guideline(GUIDE, chunk_size=200, chunk_overlap=20, version_id=vid)
    b = chunk_guideline(GUIDE, chunk_size=200, chunk_overlap=20, version_id=vid)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.document.page_content for c in a] == [c.document.page_content for c in b]
    assert a[0].id == f"{vid}:0000"
    assert a[1].id == make_chunk_id(vid, 1) == f"{vid}:0001"
    assert all(c.document.id == c.id for c in a)
    assert all(c.document.metadata["chunk_index"] == i for i, c in enumerate(a))


def test_base_metadata_propagated() -> None:
    chunks = chunk_guideline(
        GUIDE,
        chunk_size=300,
        chunk_overlap=30,
        version_id="v",
        base_metadata={"company_slug": "bookshop", "version_number": 3},
    )
    assert all(c.document.metadata["company_slug"] == "bookshop" for c in chunks)
    assert all(c.document.metadata["version_number"] == 3 for c in chunks)


def test_overlap_respected() -> None:
    # One long paragraph with no headings or blank lines: splits fall back to sentences/words.
    sentences = [f"Sentence number {i} describes policy item {i} in detail." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_guideline(text, chunk_size=200, chunk_overlap=60, version_id="v")
    assert len(chunks) > 3
    for prev, nxt in itertools.pairwise(chunks):
        p, n = prev.document.page_content, nxt.document.page_content
        assert len(p) <= 200
        # Some tail of the previous chunk re-appears at the start of the next one.
        overlap = max(k for k in range(0, min(len(p), len(n)) + 1) if p.endswith(n[:k]))
        assert overlap > 0, (p, n)
        assert overlap <= 60 + 5


def test_normalize_text() -> None:
    assert normalize_text("a  \r\nb\t\r\n\r\nc   ") == "a\nb\n\nc"
