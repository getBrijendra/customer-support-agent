"""Opt-in live smoke test: real Gemini embeddings + LLM against local Supabase.

Skipped unless GOOGLE_API_KEY is set (env or .env), the bookshop seed file exists and the
integration database is reachable. Run with: pytest -m live
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.main import create_app
from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS, make_settings, running_client
from tests.integration.conftest import create_company, pg_url  # noqa: F401  (fixture)

SEED = Path(__file__).resolve().parents[2] / "data" / "seed_guidelines" / "bookshop.txt"


@pytest.fixture
def live_settings(pg_url: str, tmp_path: Path) -> Settings:  # noqa: F811
    env = Settings()
    if not env.GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY not set")
    if not SEED.is_file():
        pytest.skip(f"seed file missing: {SEED}")
    return make_settings(
        DATABASE_URL=pg_url,
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        GOOGLE_API_KEY=env.GOOGLE_API_KEY.get_secret_value(),
        LLM_MODEL=env.LLM_MODEL,
        EMBEDDING_MODEL=env.EMBEDDING_MODEL,
        CHUNK_SIZE=env.CHUNK_SIZE,
        CHUNK_OVERLAP=env.CHUNK_OVERLAP,
        RAG_TOP_K=env.RAG_TOP_K,
    )


async def test_live_bookshop_question(live_settings: Settings) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(live_settings.DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chats, guideline_versions, companies CASCADE"))
    await engine.dispose()

    app = create_app(live_settings)
    async with running_client(app) as client:
        await create_company(client, "bookshop", "Bookshop")
        r = await client.post(
            "/api/v1/companies/bookshop/guidelines",
            files={"file": ("bookshop.txt", SEED.read_bytes(), "text/plain")},
            headers=ADMIN_HEADERS,
        )
        assert r.status_code == 201, r.text
        assert r.json()["status"] == "ready", r.json()["error_message"]

        r = await client.post(
            "/api/v1/chats",
            json={
                "company": "bookshop",
                "customer_message": "My book arrived with a torn cover. Can I get a refund?",
                "channel": "email",
            },
            headers=AGENT_HEADERS,
            timeout=180,
        )
        assert r.status_code == 201, r.text
        g = r.json()["guidance"]
        assert g["recommended_response"].strip()
        assert g["confidence"] in {"high", "medium", "low"}
        assert all(ref.split(":")[0] for ref in g["guideline_references"])
