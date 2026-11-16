import pytest

from app.core.config import Settings
from app.core.logging import request_id_var
from app.core.sentry import _before_send, _level, init_sentry

pytestmark = pytest.mark.unit


def test_disabled_without_dsn() -> None:
    assert init_sentry(Settings(SENTRY_DSN=None)) is False


def test_before_send_strips_bodies_and_tags_request_id() -> None:
    token = request_id_var.set("req-123")
    try:
        event = {
            "request": {
                "data": {"customer_message": "secret"},
                "cookies": {"a": "b"},
                "query_string": "key=admin-key",
                "url": "http://x/api/v1/chats",
            }
        }
        out = _before_send(event, {})
    finally:
        request_id_var.reset(token)
    assert out is not None
    assert out["tags"]["request_id"] == "req-123"
    assert out["request"] == {"url": "http://x/api/v1/chats"}


def test_level_parsing() -> None:
    assert _level("error") == 40
    with pytest.raises(ValueError):
        _level("loud")
