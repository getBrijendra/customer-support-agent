"""Create the demo companies, upload + compile their guideline files through the service
layer, and (optionally) bind capabilities so the demo shows verified lookups: mock stores for
bookshop / vpn-saas, and the in-process AgriFarm MCP server (`inproc://agrifarm`) for agrifarm.

Idempotent: existing companies are reused; an upload identical to the active version is skipped;
bindings are re-applied.

Usage:
  python scripts/seed.py                    # use the provider configured by LLM_PROVIDER
  python scripts/seed.py --fake-compiler    # offline: heuristic compiler/examples/router + fake
                                            # embeddings (no key needed; for demos/dev only)
  python scripts/seed.py --no-bindings      # skip the mock capability bindings
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.errors import CompanyAlreadyExists, DuplicateGuidelines  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.resources import build_resources  # noqa: E402

SEED_DIR = ROOT / "data" / "seed_guidelines"
COMPANIES = [
    ("bookshop", "Bookshop", "Online bookshop", "bookshop.txt"),
    ("vpn-saas", "VPN SaaS", "Consumer VPN subscription service", "vpn_saas.txt"),
    (
        "agrifarm",
        "AgriFarm",
        "Farm-fresh produce delivery and Farm Box subscriptions",
        "agrifarm.txt",
    ),
]
# (capability, mock store, policy, sample args for the test call)
MOCK_BINDINGS: dict[str, list[tuple[str, str, dict[str, Any], dict[str, Any]]]] = {
    "bookshop": [
        ("order.get", "bookshop_orders", {}, {"order_id": "1001"}),
        ("shipment.track", "bookshop_orders", {}, {"order_id": "1001"}),
        ("refund.create", "bookshop_orders", {"max_amount": 50}, {"amount": 10}),
        ("replacement.create", "bookshop_orders", {}, {"order_id": "1001"}),
        ("discount.issue", "bookshop_orders", {"max_pct": 10}, {"percent": 5}),
    ],
    "vpn-saas": [
        ("account.get", "vpn_accounts", {}, {"email": "jane@example.com"}),
        ("subscription.get", "vpn_accounts", {}, {"email": "jane@example.com"}),
        ("device.list", "vpn_accounts", {}, {"email": "jane@example.com"}),
        ("status.get", "vpn_accounts", {}, {}),
        ("refund.create", "vpn_accounts", {"max_amount": 100}, {"amount": 10}),
        ("password.reset", "vpn_accounts", {}, {"email": "jane@example.com"}),
    ],
}


# AgriFarm: every capability is served by the in-process MCP server (data/mcp/agrifarm.json).
AGRIFARM_MCP = {"name": "agrifarm-systems", "url": "inproc://agrifarm"}
_P = "priya.sharma@example.com"
# (capability, MCP tool, policy, sample args). Writes get the idempotency key as a tool arg.
MCP_BINDINGS: dict[str, list[tuple[str, str, dict[str, Any], dict[str, Any]]]] = {
    "agrifarm": [
        ("order.get", "get_order", {}, {"order_id": "AF-10421"}),
        ("shipment.track", "track_delivery", {}, {"order_id": "AF-10463"}),
        ("delivery_slot.list", "list_delivery_slots", {}, {"order_id": "AF-10463"}),
        ("product.availability", "check_product_availability", {}, {"product": "strawberries"}),
        ("farmbox.get", "get_farm_box", {}, {"email": _P}),
        (
            "refund.create",
            "issue_refund",
            {"max_amount": 60},
            {"order_id": "AF-10421", "amount": 9},
        ),
        ("replacement.create", "create_replacement_delivery", {}, {"order_id": "AF-10421"}),
        ("discount.issue", "issue_discount_code", {"max_pct": 15}, {"email": _P, "percent": 10}),
        (
            "delivery.reschedule",
            "reschedule_delivery",
            {},
            {"order_id": "AF-10463", "slot_id": "N-2026-11-19-AM"},
        ),
        ("order.cancel", "cancel_order", {}, {"order_id": "AF-10458"}),
        ("credit.issue", "issue_store_credit", {"max_amount": 15}, {"email": _P, "amount": 4.9}),
        ("farmbox.pause", "pause_farm_box", {"max_weeks": 4}, {"email": _P, "weeks": 2}),
        (
            "restock_alert.create",
            "create_restock_alert",
            {},
            {"email": _P, "product": "strawberries"},
        ),
    ],
}


def _overrides(fake: bool, settings: Settings) -> tuple[Settings, dict[str, Any]]:
    """--fake-compiler swaps the compiler/example generator/router for heuristic versions.

    Embeddings follow LLM_PROVIDER so the seeded index matches the running app. In Google mode
    without a key, the command falls back to LangChain's deterministic test embeddings.
    """
    if not fake:
        return settings, {}
    from app.ai.heuristic import (
        HeuristicExampleGenerator,
        HeuristicIntentSelector,
        HeuristicPlaybookCompiler,
    )

    overrides: dict[str, Any] = {
        "compiler": HeuristicPlaybookCompiler(),
        "examples": HeuristicExampleGenerator(),
        "selector": HeuristicIntentSelector(),
    }
    if settings.LLM_PROVIDER != "fake" and not settings.GOOGLE_API_KEY:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        emb = DeterministicFakeEmbedding(size=settings.EMBEDDING_DIMENSIONS)
        overrides["embeddings_for"] = lambda model: emb
        settings = settings.model_copy(update={"EMBEDDING_MODEL": "fake-embedding"})
    return settings, overrides


async def _bind_mcp(integrations: Any, slug: str) -> None:
    """Register (or reuse) the company's MCP server, discover its tools, bind + test + go live."""
    from app.capabilities.registry import get_capability

    servers = {s.name: s for s in await integrations.list_servers(slug)}
    server = servers.get(AGRIFARM_MCP["name"]) or await integrations.create_server(
        slug, name=AGRIFARM_MCP["name"], url=AGRIFARM_MCP["url"], auth_secret_name=None
    )
    server = await integrations.discover(slug, server.id)
    tools = {t["name"] for t in server.discovered_tools or []}
    print(f"[{slug}] MCP server {server.url}: {len(tools)} tools discovered")
    for cap, tool, policy, sample in MCP_BINDINGS[slug]:
        if tool not in tools:
            print(f"[{slug}] binding {cap}: FAILED (tool {tool} not on server)")
            continue
        config: dict[str, Any] = {"server_id": str(server.id), "tool_name": tool, "arg_map": {}}
        capability = get_capability(cap)
        if capability is not None and capability.kind == "write":
            config["idempotency_arg"] = "idempotency_key"
        await integrations.put_binding(slug, cap, adapter_type="mcp", config=config, policy=policy)
        _, result = await integrations.test_binding(slug, cap, sample, execute_write=False)
        if result["ok"]:
            await integrations.go_live(slug, cap)
        status = "live" if result["ok"] else f"FAILED: {result.get('error')}"
        print(f"[{slug}] binding {cap} -> mcp:{tool}: {status}")


async def main(fake: bool, bindings: bool) -> int:
    settings = get_settings()
    configure_logging("WARNING")
    missing = [f for *_, f in COMPANIES if not (SEED_DIR / f).is_file()]
    if missing:
        print(f"Missing seed guideline files in {SEED_DIR}: {', '.join(missing)}")
        print("Place the provided files there and re-run `make seed`.")
        return 1
    if fake or settings.LLM_PROVIDER == "fake":
        print("Using the OFFLINE heuristic compiler and fake embeddings (demo only).")

    settings, overrides = _overrides(fake, settings)
    resources = build_resources(settings, **overrides)
    companies = resources.company_service()
    ingestion = resources.ingestion_service()
    integrations = resources.integration_service()
    exit_code = 0
    try:
        for slug, name, description, filename in COMPANIES:
            try:
                await companies.create(slug=slug, name=name, description=description)
                print(f"[{slug}] company created")
            except CompanyAlreadyExists:
                print(f"[{slug}] company already exists")

            content = (SEED_DIR / filename).read_bytes()
            try:
                created = await ingestion.create_version(
                    slug=slug,
                    filename=filename,
                    content=content,
                    notes="seed",
                    uploaded_by="seed-script",
                )
                version = await ingestion.run_ingestion(created.version.id)
                if version is None or version.status != "ready":
                    err = version.error_message if version else "version missing"
                    print(f"[{slug}] v{created.version.version_number} FAILED: {err}")
                    exit_code = 1
                    continue
                print(
                    f"[{slug}] v{version.version_number} ready: {version.intent_count} intents, "
                    f"{version.fact_count} facts, review={version.review_status} "
                    f"({version.review_note})"
                )
            except DuplicateGuidelines as exc:
                print(f"[{slug}] skipped upload: {exc.message}")

            if bindings and slug in MCP_BINDINGS:
                await _bind_mcp(integrations, slug)
            if bindings:
                for cap, store, policy, sample in MOCK_BINDINGS.get(slug, []):
                    await integrations.put_binding(
                        slug, cap, adapter_type="mock", config={"store": store}, policy=policy
                    )
                    _, result = await integrations.test_binding(
                        slug, cap, sample, execute_write=False
                    )
                    if result["ok"]:
                        await integrations.go_live(slug, cap)
                    print(f"[{slug}] binding {cap}: {'live' if result['ok'] else 'FAILED'}")
    finally:
        await resources.aclose()
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fake-compiler", action="store_true", help="offline heuristic mode")
    parser.add_argument("--no-bindings", action="store_true", help="skip mock bindings")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.fake_compiler, not args.no_bindings)))
