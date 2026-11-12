"""U6 — Chroma collection naming."""

from __future__ import annotations

import uuid

from app.vectorstore.base import CHROMA_NAME_RE, make_collection_name


def test_names_are_valid_for_chroma() -> None:
    for n in (1, 2, 10, 9999):
        name = make_collection_name(uuid.uuid4(), n)
        assert CHROMA_NAME_RE.match(name), name
        assert 3 <= len(name) <= 512
        assert name.replace("_", "").isalnum()


def test_unique_per_company_and_version() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    names = {make_collection_name(c, v) for c in (a, b) for v in (1, 2, 3)}
    assert len(names) == 6


def test_deterministic() -> None:
    cid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert make_collection_name(cid, 3) == "gl_123456781234_v3"
