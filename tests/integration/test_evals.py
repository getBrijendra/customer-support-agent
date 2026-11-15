"""Integration case 21: routing evals (retrieval + full mode), manual cases, agent feedback."""

from __future__ import annotations

import httpx

from tests.conftest import ADMIN_HEADERS, AGENT_HEADERS, ScriptedSelector
from tests.integration.conftest import BOOKSHOP_TEXT, ClientFactory, chat, create_company, upload


async def test_eval_cases_runs_and_feedback(client_factory: ClientFactory) -> None:
    async with client_factory(selector=ScriptedSelector()) as client:
        await create_company(client, "bookshop")
        v1 = (await upload(client, "bookshop", BOOKSHOP_TEXT)).json()

        # Ingest generated synthetic held-out cases (3 per intent) + negatives and ran an eval.
        cases = (
            await client.get("/api/v1/companies/bookshop/eval-cases", headers=ADMIN_HEADERS)
        ).json()
        synthetic = [c for c in cases if c["source"] == "synthetic"]
        assert len([c for c in synthetic if c["expected_intent_keys"]]) == 9
        assert any(c["expected_intent_keys"] == [] for c in synthetic)  # negatives
        run_id = v1["routing_eval_run_id"]
        run = (
            await client.get(f"/api/v1/internal/evals/routing/runs/{run_id}", headers=ADMIN_HEADERS)
        ).json()
        assert run["mode"] == "retrieval" and run["version_number"] == 1
        m = run["metrics"]
        assert m["positives"] == 9 and m["negatives"] == 4
        assert m["top1_accuracy"] >= 0.5 and m["recall_at_k"] == 1.0
        assert m["no_match_recall"] == 1.0  # off-topic negatives are recognised
        assert set(m["per_intent"]) == {"shipping_delays", "damaged_books", "returns"}
        assert len(run["results"]) == m["cases"]

        # Manual case + a stale case (key that doesn't exist in this version).
        for body in (
            {"message": "my parcel never came", "expected_intent_keys": ["shipping_delays"]},
            {"message": "gift card", "expected_intent_keys": ["gift_cards"]},
        ):
            r = await client.post(
                "/api/v1/companies/bookshop/eval-cases", json=body, headers=ADMIN_HEADERS
            )
            assert r.status_code == 201 and r.json()["source"] == "manual"

        r = await client.post(
            "/api/v1/internal/evals/routing",
            json={"company": "bookshop", "mode": "full"},
            headers=ADMIN_HEADERS,
        )
        assert r.status_code == 200, r.text
        full = r.json()
        assert full["mode"] == "full" and full["metrics"]["stale"] == 1
        assert full["metrics"]["cases"] == m["cases"] + 2

        runs = (
            await client.get(
                "/api/v1/internal/evals/routing/runs",
                params={"company": "bookshop"},
                headers=ADMIN_HEADERS,
            )
        ).json()
        assert [x["mode"] for x in runs] == ["full", "retrieval"] and runs[0]["results"] is None

        # Agent feedback corrects the routing and becomes an eval case.
        c = (await chat(client, "bookshop", "My parcel is late and the cover is torn")).json()
        r = await client.post(
            f"/api/v1/chats/{c['chat_id']}/feedback",
            json={"correct_intent_keys": ["shipping_delays", "damaged_books"]},
            headers=AGENT_HEADERS,
        )
        assert r.status_code == 200 and r.json()["changed"] is True
        detail = (await client.get(f"/api/v1/chats/{c['chat_id']}", headers=AGENT_HEADERS)).json()
        assert detail["feedback"]["correct_intent_keys"] == ["shipping_delays", "damaged_books"]
        cases = (
            await client.get("/api/v1/companies/bookshop/eval-cases", headers=ADMIN_HEADERS)
        ).json()
        fb = [x for x in cases if x["source"] == "agent_feedback"]
        assert len(fb) == 1 and fb[0]["chat_id"] == c["chat_id"]


async def test_eval_errors(client: httpx.AsyncClient) -> None:
    await create_company(client, "bookshop")
    r = await client.post(
        "/api/v1/internal/evals/routing", json={"company": "bookshop"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "NO_ACTIVE_GUIDELINES"
    r = await client.post(
        "/api/v1/internal/evals/routing",
        json={"company": "bookshop", "version_number": 7},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 404
    r = await client.get(
        "/api/v1/internal/evals/routing/runs/00000000-0000-0000-0000-000000000000",
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 404
