"""U4 — cursor encode/decode."""

from __future__ import annotations

import base64
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.core.errors import ValidationFailed
from app.services.chat_history_service import decode_cursor, encode_cursor


def test_round_trip() -> None:
    ts = datetime(2026, 11, 10, 8, 0, 0, 123456, tzinfo=UTC)
    cid = uuid.uuid4()
    cursor = encode_cursor(ts, cid)
    assert "=" not in cursor
    assert decode_cursor(cursor) == (ts, cid)


def test_round_trip_non_utc_offset() -> None:
    ts = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    cid = uuid.uuid4()
    got_ts, got_id = decode_cursor(encode_cursor(ts, cid))
    assert got_ts == ts and got_ts.utcoffset() == ts.utcoffset()
    assert got_id == cid


@pytest.mark.parametrize(
    "cursor",
    [
        "not-base64!!",
        base64.urlsafe_b64encode(b"[1,2,3]").decode(),
        base64.urlsafe_b64encode(b'["2026-01-01T00:00:00+00:00", "nope"]').decode(),
        base64.urlsafe_b64encode(
            b'["2026-01-01T00:00:00", "%s"]' % str(uuid.uuid4()).encode()
        ).decode(),
        base64.urlsafe_b64encode(b"{}").decode(),
        "",
    ],
)
def test_invalid_cursor(cursor: str) -> None:
    with pytest.raises(ValidationFailed) as exc:
        decode_cursor(cursor)
    assert exc.value.code == "VALIDATION_ERROR"
