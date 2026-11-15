"""Integration cases 1-7, 16, 17, 22: compilation, versioning, approval, activation, failures,
duplicates, concurrency, stable keys, cascades."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS
from tests.integration.conftest import (
    BOOKSHOP_TEXT,
    ClientFactory,
    active_version_ids,
    app_of,
    chat,
    create_company,
    upload,
)

BASE = "/api/v1/companies/bookshop/guidelines"


async def _versions(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(BASE, headers=AGENT_HEADERS)
    assert r.status_code == 200
    return r.json()  # type: ignore[no-any-return]


async def _assert_single_active(
    client: httpx.AsyncClient, sm: async_sessionmaker[AsyncSession], expected_number: int | None
) -> None:
    """Case 4: exactly one active version per company, matching the pointer in the DB."""
    versions = await _versions(client)
    active = [v for v in versions if v["is_active"]]
    pointer = await active_version_ids(sm, "bookshop")
    assert len(pointer) == 1
    if expected_number is None:
        assert active == [] and pointer == [None]
    else:
        assert len(active) == 1 and active[0]["version_number"] == expected_number
        assert str(pointer[0]) == active[0]["id"]


async def _count(sm: async_sessionmaker[AsyncSession], table: str, version_id: str) -> int:
    async with sm() as s:
        return int(
            (
                await s.execute(
                    text(f"SELECT count(*) FROM {table} WHERE version_id = :v"), {"v": version_id}
                )
            ).scalar_one()
        )


async def test_upload_compiles_v1_and_activates(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 1."""
    await create_company(client, "bookshop")
    r = await upload(client, "bookshop", BOOKSHOP_TEXT, notes="initial import")
    assert r.status_code == 201, r.text  # INGESTION_SYNC=true in tests
    v1 = r.json()
    assert v1["version_number"] == 1 and v1["status"] == "ready", v1["error_message"]
    assert v1["review_status"] == "approved" and v1["reviewed_by"] == "auto" and v1["is_active"]
    assert v1["section_count"] == 5 and v1["intent_count"] == 3 and v1["fact_count"] == 2
    assert v1["coverage"]["unmapped"] == [] and v1["routing_eval_run_id"]
    assert v1["notes"] == "initial import" and v1["uploaded_by"] == "tester"
    assert v1["embedding_model"] == "fake-embedding" and v1["embedding_dimensions"] == 768
    assert v1["compiler_model"] == "heuristic-compiler" and v1["compile_prompt_version"]
    for table, n in [
        ("guideline_sections", 5),
        ("intents", 3),
        ("company_rules", 1),
        ("reference_facts", 2),
    ]:
        assert await _count(db_sessionmaker, table, v1["id"]) == n, table
    assert await _count(db_sessionmaker, "intent_examples", v1["id"]) > 3
    async with db_sessionmaker() as s:
        nulls = (
            await s.execute(
                text(
                    "SELECT count(*) FROM intent_examples WHERE split='index' AND embedding IS NULL"
                )
            )
        ).scalar_one()
    assert nulls == 0
    await _assert_single_active(client, db_sessionmaker, 1)

    pb = (await client.get(f"{BASE}/1/playbook", headers=AGENT_HEADERS)).json()
    keys = [i["key"] for i in pb["intents"]]
    assert keys == ["damaged_books", "returns", "shipping_delays"]
    damaged = next(i for i in pb["intents"] if i["key"] == "damaged_books")
    assert [s["type"] for s in damaged["steps"]] == ["ask", "lookup", "action"]
    assert damaged["steps"][2]["capability"] == "refund.create"
    assert "Warm regards" in pb["company_rules"]["sign_off"]
    assert pb["company_rules"]["prohibitions"]
    assert {f["title"] for f in pb["facts"]} == {"Standard shipping", "Express shipping"}

    r = await client.get(f"{BASE}/1", params={"include_text": "true"}, headers=AGENT_HEADERS)
    assert r.json()["raw_text"].startswith("TONE")
    detail = (await client.get("/api/v1/companies/bookshop", headers=AGENT_HEADERS)).json()
    assert detail["active_version_number"] == 1


async def test_second_upload_activation_and_rollback(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Cases 2, 3, 4 and 17 (rollback restores the old playbook + rules)."""
    await create_company(client, "bookshop")
    v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    await _assert_single_active(client, db_sessionmaker, 1)

    v2_text = BOOKSHOP_TEXT.replace("Warm regards", "Kind regards") + (
        "\n4. Gift Cards\nApproach to solve:\n- Verify the gift card code.\n"
    )
    v2 = (await upload(client, "bookshop", v2_text)).json()
    assert v2["version_number"] == 2 and v2["is_active"] and v2["intent_count"] == 4
    versions = await _versions(client)
    assert [v["version_number"] for v in versions] == [2, 1]
    assert versions[1]["status"] == "ready" and not versions[1]["is_active"]
    await _assert_single_active(client, db_sessionmaker, 2)

    # Case 3: roll back to v1; the next chat uses v1's intents AND v1's company rules.
    r = await client.post(f"{BASE}/1/activate", headers=ADMIN_HEADERS)
    assert r.status_code == 200 and r.json()["is_active"]
    await _assert_single_active(client, db_sessionmaker, 1)
    r = await chat(client, "bookshop", "My gift cards code does not work")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["guideline_version"] == 1
    detail = (await client.get(f"/api/v1/chats/{body['chat_id']}", headers=AGENT_HEADERS)).json()
    assert detail["guideline_version_id"] == v1["id"]
    assert "gift_cards" not in [c["key"] for c in detail["routing"]["candidates"]]

    gen = app_of(client).state.resources.generator
    assert "Warm regards" in gen.calls[-1]["context"].rules["sign_off"]

    # Roll forward again.
    assert (await client.post(f"{BASE}/2/activate", headers=ADMIN_HEADERS)).status_code == 200
    await _assert_single_active(client, db_sessionmaker, 2)
    await chat(client, "bookshop", "gift cards")
    assert "Kind regards" in gen.calls[-1]["context"].rules["sign_off"]

    # Upload without activation keeps v2 active (but is approved and can be activated later).
    v3 = (await upload(client, "bookshop", v2_text + "\nX:\ny\n", activate=False)).json()
    assert v3["status"] == "ready" and v3["review_status"] == "approved" and not v3["is_active"]
    await _assert_single_active(client, db_sessionmaker, 2)


async def test_manual_approval_workflow(
    client_factory: ClientFactory, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 16: auto-approval off -> pending, not active; approve -> active; reject blocks."""
    async with client_factory(AUTO_APPROVE_GUIDELINES=False) as client:
        await create_company(client, "bookshop")
        v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
        assert v1["status"] == "ready" and v1["review_status"] == "pending"
        assert not v1["is_active"] and "disabled" in v1["review_note"]
        await _assert_single_active(client, db_sessionmaker, None)
        assert (await chat(client, "bookshop")).json()["error"]["code"] == "NO_ACTIVE_GUIDELINES"

        r = await client.post(f"{BASE}/1/activate", headers=ADMIN_HEADERS)
        assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_NOT_APPROVED"

        r = await client.post(f"{BASE}/1/approve", json={"note": "lgtm"}, headers=ADMIN_HEADERS)
        assert r.status_code == 200, r.text
        assert r.json()["review_status"] == "approved" and r.json()["is_active"]
        assert r.json()["reviewed_by"] == "tester"
        await _assert_single_active(client, db_sessionmaker, 1)
        assert (await chat(client, "bookshop")).status_code == 201
        r = await client.post(f"{BASE}/1/approve", headers=ADMIN_HEADERS)
        assert r.status_code == 409

        await upload(client, "bookshop", BOOKSHOP_TEXT + "\nNEW:\nx\n")
        r = await client.post(f"{BASE}/2/reject", json={"reason": "typo"}, headers=ADMIN_HEADERS)
        assert r.status_code == 200 and r.json()["review_status"] == "rejected"
        r = await client.post(f"{BASE}/2/activate", headers=ADMIN_HEADERS)
        assert r.json()["error"]["code"] == "VERSION_NOT_APPROVED"
        r = await client.post(f"{BASE}/1/reject", headers=ADMIN_HEADERS)  # active one
        assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_IS_ACTIVE"
        await _assert_single_active(client, db_sessionmaker, 1)


async def test_eval_gate_blocks_auto_approval(client_factory: ClientFactory) -> None:
    async with client_factory(AUTO_APPROVE_MIN_ROUTING_TOP1=1.0) as client:
        await create_company(client, "bookshop")
        # A case the router can't get right (no keyword overlap) pulls top-1 below 1.0.
        r = await client.post(
            "/api/v1/companies/bookshop/eval-cases",
            json={"message": "pizza tonight?", "expected_intent_keys": ["damaged_books"]},
            headers=ADMIN_HEADERS,
        )
        assert r.status_code == 201
        v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
        assert v1["review_status"] == "pending" and "below" in v1["review_note"]


async def test_failed_compilation_keeps_previous_active(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: Any
) -> None:
    """Case 5: a failure leaves no compiled rows and the previous version keeps serving."""
    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    emb = app_of(client).state.resources.embeddings_for("fake-embedding")

    def boom(self: Any, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding provider is down")

    monkeypatch.setattr(type(emb), "embed_documents", boom)
    r = await upload(client, "bookshop", BOOKSHOP_TEXT + "\nCHANGED:\nyes\n")
    assert r.status_code == 201
    failed = r.json()
    assert failed["version_number"] == 2 and failed["status"] == "failed"
    assert "embedding provider is down" in failed["error_message"]
    assert not failed["is_active"]
    for table in ("guideline_sections", "intents", "intent_examples", "company_rules"):
        assert await _count(db_sessionmaker, table, failed["id"]) == 0
    await _assert_single_active(client, db_sessionmaker, 1)

    r = await client.post(f"{BASE}/2/activate", headers=ADMIN_HEADERS)
    assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_NOT_READY"
    monkeypatch.undo()
    r = await chat(client, "bookshop", "hi")
    assert r.status_code == 201 and r.json()["guideline_version"] == 1


async def test_compile_error_without_intents(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    r = await upload(client, "bookshop", "TONE\nBe nice.\n")
    v = r.json()
    assert v["status"] == "failed" and "No intents" in v["error_message"]


async def test_duplicate_upload_of_active_content_409(client: httpx.AsyncClient) -> None:
    """Case 6."""
    await create_company(client, "bookshop")
    await upload(client, "bookshop", BOOKSHOP_TEXT)
    r = await upload(client, "bookshop", BOOKSHOP_TEXT, filename="again.txt")
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "DUPLICATE_GUIDELINES"
    assert err["details"]["existing_version"]["version_number"] == 1
    assert len(await _versions(client)) == 1

    await upload(client, "bookshop", BOOKSHOP_TEXT + "\nV2:\nx\n")
    r = await upload(client, "bookshop", BOOKSHOP_TEXT)
    assert r.status_code == 201
    assert r.json()["version_number"] == 3 and "v1" in r.json()["info"]


async def test_concurrent_uploads_get_distinct_versions(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    """Case 7: SELECT ... FOR UPDATE serialises version allocation."""
    await create_company(client, "bookshop")
    results = await asyncio.gather(
        *(upload(client, "bookshop", BOOKSHOP_TEXT + f"\nVARIANT {i}:\nx{i}\n") for i in range(4))
    )
    assert all(r.status_code == 201 for r in results), [r.text for r in results]
    numbers = sorted(r.json()["version_number"] for r in results)
    assert numbers == [1, 2, 3, 4]
    await _assert_single_active(client, db_sessionmaker, 4)  # latest wins


async def test_stable_intent_keys_across_versions(client_factory: ClientFactory) -> None:
    """Case 22: the compiler gets the previous active intents and reuses their keys."""
    calls: list[list[dict[str, str]]] = []
    from app.ai.heuristic import HeuristicPlaybookCompiler

    class Spy(HeuristicPlaybookCompiler):
        async def compile(self, doc: Any, *, company_name: str, previous_intents: Any) -> Any:
            calls.append(previous_intents)
            return await super().compile(
                doc, company_name=company_name, previous_intents=previous_intents
            )

    async with client_factory(compiler=Spy()) as client:
        await create_company(client, "bookshop")
        await upload(client, "bookshop", BOOKSHOP_TEXT)
        await upload(client, "bookshop", BOOKSHOP_TEXT + "\nNOTE:\nx\n")
        assert calls[0] == []
        assert {p["key"] for p in calls[1]} == {"shipping_delays", "damaged_books", "returns"}
        pb1 = (await client.get(f"{BASE}/1/playbook", headers=AGENT_HEADERS)).json()
        pb2 = (await client.get(f"{BASE}/2/playbook", headers=AGENT_HEADERS)).json()
        assert {i["key"] for i in pb1["intents"]} <= {i["key"] for i in pb2["intents"]}


async def test_delete_version_cascades(
    client: httpx.AsyncClient, db_sessionmaker: async_sessionmaker[AsyncSession]
) -> None:
    await create_company(client, "bookshop")
    v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()
    await upload(client, "bookshop", BOOKSHOP_TEXT + "\nMORE:\nx\n")
    r = await client.delete(f"{BASE}/2", headers=ADMIN_HEADERS)
    assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_IS_ACTIVE"
    assert await _count(db_sessionmaker, "intent_examples", v1["id"]) > 0
    r = await client.delete(f"{BASE}/1", headers=ADMIN_HEADERS)
    assert r.status_code == 204
    for table in ("guideline_sections", "intents", "intent_examples", "reference_facts"):
        assert await _count(db_sessionmaker, table, v1["id"]) == 0
    assert [v["version_number"] for v in await _versions(client)] == [2]
    assert (await client.get(f"{BASE}/1", headers=AGENT_HEADERS)).status_code == 404


async def test_upload_validation_errors(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    r = await upload(client, "bookshop", "hello", filename="guide.pdf")
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"
    r = await upload(client, "bookshop", "   ")
    assert r.json()["error"]["code"] == "INVALID_FILE"
    r = await upload(client, "nope", BOOKSHOP_TEXT)
    assert r.status_code == 404 and r.json()["error"]["code"] == "COMPANY_NOT_FOUND"


async def test_upload_too_large(client_factory: ClientFactory) -> None:
    async with client_factory(MAX_UPLOAD_BYTES=100) as client:
        await create_company(client, "bookshop")
        r = await upload(client, "bookshop", "x" * 101)
        assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


async def test_async_mode_returns_202_then_ready(client_factory: ClientFactory) -> None:
    async with client_factory(INGESTION_SYNC=False) as client:
        await create_company(client, "bookshop")
        r = await upload(client, "bookshop", BOOKSHOP_TEXT)
        assert r.status_code == 202
        assert r.json()["status"] == "processing" and r.json()["review_status"] == "pending"
        # httpx's ASGI transport runs background tasks before returning the response
        v = (await client.get(f"{BASE}/1", headers=AGENT_HEADERS)).json()
        assert v["status"] == "ready" and v["is_active"]
