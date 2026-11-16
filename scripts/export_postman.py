"""Generate a Postman (v2.1) collection + environment for every API endpoint.

The requests are curated (realistic bodies, query params, chaining scripts that save ids into
collection variables), and the script fails if any route in the app's OpenAPI schema is not
covered, so the collection can't silently drift from the API.

Usage: python scripts/export_postman.py   -> postman/*.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "postman"
API = "{{baseUrl}}/api/v1"
ADMIN = "admin"
AGENT = "agent"
PUBLIC = None


def save(var: str, expr: str) -> str:
    """JS line that stores a value from the JSON response into a collection variable."""
    return (
        f"try {{ const v = {expr}; if (v) pm.collectionVariables.set('{var}', v); }} "
        f"catch (e) {{ console.log('could not save {var}', e); }}"
    )


def req(
    name: str,
    method: str,
    path: str,
    *,
    auth: str | None = AGENT,
    body: Any = None,
    query: dict[str, str] | None = None,
    form: list[dict[str, Any]] | None = None,
    desc: str = "",
    tests: list[str] | None = None,
    base: str = API,
) -> dict[str, Any]:
    headers = []
    if auth:
        headers.append({"key": "X-API-Key", "value": f"{{{{{auth}_key}}}}", "type": "text"})
    url_raw = base + path
    url: dict[str, Any] = {
        "raw": url_raw + ("?" + "&".join(f"{k}={v}" for k, v in query.items()) if query else ""),
        "host": ["{{baseUrl}}"],
        "path": [p for p in (url_raw.replace("{{baseUrl}}", "")).split("/") if p],
    }
    if query:
        url["query"] = [{"key": k, "value": v} for k, v in query.items()]
    request: dict[str, Any] = {
        "method": method,
        "header": headers,
        "url": url,
        "description": desc,
    }
    if body is not None:
        headers.append({"key": "Content-Type", "value": "application/json", "type": "text"})
        request["body"] = {
            "mode": "raw",
            "raw": json.dumps(body, indent=2),
            "options": {"raw": {"language": "json"}},
        }
    if form is not None:
        request["body"] = {"mode": "formdata", "formdata": form}
    item: dict[str, Any] = {"name": name, "request": request}
    checks = [
        "pm.test('status is 2xx', () => pm.expect(pm.response.code).to.be.within(200, 299));",
        *(tests or []),
    ]
    item["event"] = [{"listen": "test", "script": {"type": "text/javascript", "exec": checks}}]
    return item


def folder(name: str, desc: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "description": desc, "item": items}


CO = "/companies/{{company}}"
GL = CO + "/guidelines"
BIND = CO + "/capability-bindings"

ITEMS = [
    folder(
        "Health",
        "Public liveness/readiness probes.",
        [
            req("Liveness", "GET", "/health", auth=PUBLIC),
            req("Readiness (DB + pgvector)", "GET", "/ready", auth=PUBLIC),
        ],
    ),
    folder(
        "Companies",
        "Tenants. Create is admin-only; reads need an agent key.",
        [
            req(
                "Create company (bookshop)",
                "POST",
                "/companies",
                auth=ADMIN,
                body={"slug": "bookshop", "name": "Bookshop", "description": "Online bookshop"},
                desc="201, or 409 COMPANY_EXISTS if it already exists.",
            ),
            req(
                "Create company (vpn-saas)",
                "POST",
                "/companies",
                auth=ADMIN,
                body={
                    "slug": "vpn-saas",
                    "name": "VPN SaaS",
                    "description": "Consumer VPN subscription service",
                },
            ),
            req("List companies", "GET", "/companies"),
            req("Get company", "GET", CO),
        ],
    ),
    folder(
        "Guidelines",
        "Versioned uploads, compiled playbooks, review and activation. `{{company}}` selects "
        "the tenant; `{{version}}` the version number.",
        [
            req(
                "Upload guidelines (.txt)",
                "POST",
                GL,
                auth=ADMIN,
                form=[
                    {
                        "key": "file",
                        "type": "file",
                        "src": "data/seed_guidelines/bookshop.txt",
                        "description": "Pick the .txt file (Postman needs a path in its "
                        "working directory or an absolute path).",
                    },
                    {"key": "activate", "value": "true", "type": "text"},
                    {"key": "notes", "value": "refund policy updated", "type": "text"},
                ],
                desc="202 (background) or 201 when INGESTION_SYNC=true. Parses, compiles, "
                "indexes, evaluates; auto-approves and activates by default. 409 "
                "DUPLICATE_GUIDELINES if identical to the active version; 422 "
                "UNSUPPORTED_FORMAT for non-.txt.",
                tests=[save("version", "pm.response.json().version_number")],
            ),
            req("List versions (newest first)", "GET", GL),
            req("Active version", "GET", GL + "/active", query={"include_text": "false"}),
            req(
                "Get version (with raw text)",
                "GET",
                GL + "/{{version}}",
                query={"include_text": "true"},
            ),
            req(
                "Download original uploaded file",
                "GET",
                GL + "/{{version}}/file",
                desc="Returns the exact uploaded bytes from object storage (Supabase "
                "Storage) as an attachment. 404 FILE_NOT_STORED when file storage was "
                "disabled at upload time; 502 STORAGE_ERROR when storage is unreachable. "
                "Use Postman's 'Send and Download' to save it.",
            ),
            req(
                "Compiled playbook (intents, rules, facts, coverage)",
                "GET",
                GL + "/{{version}}/playbook",
            ),
            req(
                "Capability coverage (live vs manual)",
                "GET",
                GL + "/{{version}}/capability-coverage",
                auth=ADMIN,
            ),
            req(
                "Approve version",
                "POST",
                GL + "/{{version}}/approve",
                auth=ADMIN,
                body={"note": "Reviewed playbook, looks good"},
                desc="Only ready + pending/rejected versions; activates if uploaded with "
                "activate=true. 409 if already approved.",
            ),
            req(
                "Reject version",
                "POST",
                GL + "/{{version}}/reject",
                auth=ADMIN,
                body={"reason": "Refund section compiled incorrectly"},
                desc="Not allowed on the active version (409 VERSION_IS_ACTIVE).",
            ),
            req(
                "Activate version (rollback / roll forward)",
                "POST",
                GL + "/{{version}}/activate",
                auth=ADMIN,
                desc="Only ready + approved (409 VERSION_NOT_READY / VERSION_NOT_APPROVED).",
            ),
            req(
                "Delete version",
                "DELETE",
                GL + "/{{version}}",
                auth=ADMIN,
                desc="204. Refuses the active version (409). Compiled rows cascade.",
            ),
        ],
    ),
    folder(
        "Integrations",
        "Admin: capability registry, company secrets, MCP servers and capability bindings "
        "(draft -> test -> go-live).",
        [
            req("Capability registry", "GET", "/capabilities", auth=ADMIN),
            req(
                "Create / replace secret",
                "POST",
                CO + "/secrets",
                auth=ADMIN,
                body={"name": "{{secret_name}}", "value": "sk-live-replace-me"},
                desc="Encrypted at rest; the value is never returned.",
            ),
            req("List secret names", "GET", CO + "/secrets", auth=ADMIN),
            req("Delete secret", "DELETE", CO + "/secrets/{{secret_name}}", auth=ADMIN),
            req(
                "Register MCP server",
                "POST",
                CO + "/mcp-servers",
                auth=ADMIN,
                body={
                    "name": "orders",
                    "url": "https://mcp.bookshop.example/mcp",
                    "auth_secret_name": "{{secret_name}}",
                },
                tests=[save("server_id", "pm.response.json().id")],
            ),
            req("List MCP servers", "GET", CO + "/mcp-servers", auth=ADMIN),
            req(
                "Update MCP server",
                "PATCH",
                CO + "/mcp-servers/{{server_id}}",
                auth=ADMIN,
                body={"url": "https://mcp.bookshop.example/mcp/v2"},
            ),
            req(
                "Discover MCP tools",
                "POST",
                CO + "/mcp-servers/{{server_id}}/discover",
                auth=ADMIN,
                desc="Connects to the server and stores its tool list (502 TOOL_ERROR if "
                "unreachable).",
            ),
            req("Delete MCP server", "DELETE", CO + "/mcp-servers/{{server_id}}", auth=ADMIN),
            req(
                "Bind capability: mock (order.get)",
                "PUT",
                BIND + "/order.get",
                auth=ADMIN,
                body={"adapter_type": "mock", "config": {"store": "bookshop_orders"}, "policy": {}},
                desc="Creates/updates the binding as draft. Stores: bookshop_orders | "
                "vpn_accounts.",
            ),
            req(
                "Bind capability: mock write with policy (refund.create)",
                "PUT",
                BIND + "/refund.create",
                auth=ADMIN,
                body={
                    "adapter_type": "mock",
                    "config": {"store": "bookshop_orders"},
                    "policy": {"max_amount": 50},
                },
            ),
            req(
                "Bind capability: HTTP API (order.get)",
                "PUT",
                BIND + "/order.get",
                auth=ADMIN,
                body={
                    "adapter_type": "http",
                    "config": {
                        "method": "GET",
                        "url": "https://api.bookshop.example/orders/{order_id}",
                        "headers": {"Accept": "application/json"},
                        "query": {"email": "{email}"},
                        "auth_secret": "{{secret_name}}",
                        "auth_header": "Authorization",
                        "auth_scheme": "Bearer",
                        "response_path": "data",
                    },
                    "policy": {},
                },
            ),
            req(
                "Bind capability: read-only SQL (shipment.track)",
                "PUT",
                BIND + "/shipment.track",
                auth=ADMIN,
                body={
                    "adapter_type": "sql_readonly",
                    "config": {
                        "dsn_secret": "orders_db_dsn",
                        "statement": "SELECT status, carrier, eta FROM shipments "
                        "WHERE order_id = :order_id",
                    },
                    "policy": {},
                },
                desc="Only a single SELECT is accepted (422 otherwise). Runs in a READ ONLY "
                "transaction with a timeout and row limit.",
            ),
            req(
                "Bind capability: MCP tool (shipment.track)",
                "PUT",
                BIND + "/shipment.track",
                auth=ADMIN,
                body={
                    "adapter_type": "mcp",
                    "config": {
                        "server_id": "{{server_id}}",
                        "tool_name": "lookup_order",
                        "arg_map": {"order_id": "id"},
                    },
                    "policy": {},
                },
            ),
            req("List bindings", "GET", BIND, auth=ADMIN),
            req("Get binding", "GET", BIND + "/{{capability}}", auth=ADMIN),
            req(
                "Test binding (dry run)",
                "POST",
                BIND + "/{{capability}}/test",
                auth=ADMIN,
                body={"sample_args": {"order_id": "1001"}, "execute_write": False},
                desc="Reads are executed with the sample args; writes are only validated "
                "unless execute_write=true. Success moves draft -> tested.",
            ),
            req(
                "Go live",
                "POST",
                BIND + "/{{capability}}/go-live",
                auth=ADMIN,
                desc="Only from tested (409 BINDING_NOT_TESTED).",
            ),
            req("Delete binding", "DELETE", BIND + "/{{capability}}", auth=ADMIN),
        ],
    ),
    folder(
        "Chats",
        "Generate guidance, browse history, correct the routing.",
        [
            req(
                "Create chat (single intent + lookup)",
                "POST",
                "/chats",
                body={
                    "company": "bookshop",
                    "customer_message": "My order #1001 is late, where is it?",
                    "agent_notes": "Customer: maria@example.com",
                    "channel": "email",
                },
                tests=[
                    save("chat_id", "pm.response.json().chat_id"),
                    save("action_id", "(pm.response.json().proposed_actions[0] || {}).id"),
                ],
            ),
            req(
                "Create chat (multi-intent, VPN)",
                "POST",
                "/chats",
                body={
                    "company": "vpn-saas",
                    "customer_message": "I was charged twice this month and the VPN won't "
                    "connect on my iPhone.",
                    "agent_notes": "Customer email: jane@example.com, annual plan",
                    "channel": "chat",
                },
                tests=[
                    save("chat_id", "pm.response.json().chat_id"),
                    save("action_id", "(pm.response.json().proposed_actions[0] || {}).id"),
                ],
            ),
            req(
                "Create chat (refund -> proposed action)",
                "POST",
                "/chats",
                body={
                    "company": "bookshop",
                    "customer_message": "My book arrived with a torn cover. Can I get a refund?",
                    "agent_notes": "Order #1001, maria@example.com",
                    "channel": "email",
                },
                desc="With refund.create bound live, the guidance may include a proposed "
                "action; its id is saved as {{action_id}}.",
                tests=[
                    save("chat_id", "pm.response.json().chat_id"),
                    save("action_id", "(pm.response.json().proposed_actions[0] || {}).id"),
                ],
            ),
            req(
                "List chats (filters)",
                "GET",
                "/chats",
                query={
                    "company": "{{company}}",
                    "status": "success",
                    "intent": "",
                    "from": "2026-11-01T00:00:00Z",
                    "to": "2026-12-01T00:00:00Z",
                    "limit": "20",
                },
                desc="Remove empty params. `intent` filters on the primary intent key.",
                tests=[save("cursor", "pm.response.json().next_cursor")],
            ),
            req(
                "List chats (next page)",
                "GET",
                "/chats",
                query={"cursor": "{{cursor}}", "limit": "20"},
            ),
            req(
                "Get chat (routing, tool calls, actions)",
                "GET",
                "/chats/{{chat_id}}",
            ),
            req(
                "Feedback: correct the intents",
                "POST",
                "/chats/{{chat_id}}/feedback",
                body={"correct_intent_keys": ["shipping_delays"]},
                desc="Stored on the chat and added as an agent_feedback eval case.",
            ),
        ],
    ),
    folder(
        "Actions",
        "Proposed writes: the agent approves (executes exactly once) or rejects.",
        [
            req("List proposed actions", "GET", "/chats/{{chat_id}}/actions"),
            req(
                "Approve + execute action",
                "POST",
                "/chats/{{chat_id}}/actions/{{action_id}}/approve",
                desc="Re-checks binding + policy, executes with the idempotency key. "
                "Approving again returns the stored result; 409 ACTION_NOT_PENDING if denied, "
                "rejected or still executing.",
            ),
            req(
                "Reject action",
                "POST",
                "/chats/{{chat_id}}/actions/{{action_id}}/reject",
                body={"reason": "Customer accepted store credit instead"},
            ),
        ],
    ),
    folder(
        "Evals",
        "Routing eval cases and runs (admin).",
        [
            req("List eval cases", "GET", CO + "/eval-cases", auth=ADMIN),
            req(
                "Add eval case",
                "POST",
                CO + "/eval-cases",
                auth=ADMIN,
                body={
                    "message": "My parcel never arrived",
                    "expected_intent_keys": ["shipping_delays"],
                },
                desc="Empty expected_intent_keys = an out-of-scope (negative) case.",
            ),
            req(
                "Run routing eval",
                "POST",
                "/internal/evals/routing",
                auth=ADMIN,
                body={"company": "{{company}}", "version_number": None, "mode": "retrieval"},
                desc="mode: retrieval (search only) or full (adds the router LLM). "
                "version_number null = active version.",
                tests=[save("run_id", "pm.response.json().id")],
            ),
            req(
                "List eval runs",
                "GET",
                "/internal/evals/routing/runs",
                auth=ADMIN,
                query={"company": "{{company}}"},
            ),
            req(
                "Get eval run (metrics + per-case results)",
                "GET",
                "/internal/evals/routing/runs/{{run_id}}",
                auth=ADMIN,
            ),
        ],
    ),
    folder(
        "Internal",
        "Metrics (admin).",
        [
            req(
                "Metrics summary",
                "GET",
                "/internal/metrics/summary",
                auth=ADMIN,
                query={
                    "from": "2026-11-01T00:00:00Z",
                    "to": "2026-12-01T00:00:00Z",
                    "company": "{{company}}",
                },
                desc="Defaults to the last 7 days. Includes routing and tools blocks.",
            ),
            req(
                "Dashboard (HTML)",
                "GET",
                "/internal/dashboard",
                auth=ADMIN,
                desc="Open in a browser with ?key=<admin key> instead of the header.",
            ),
            req("Prometheus metrics", "GET", "/metrics", auth=ADMIN, base="{{baseUrl}}"),
        ],
    ),
]

VARIABLES = {
    "baseUrl": "http://127.0.0.1:8000",
    "agent_key": "change-me-agent-key",
    "admin_key": "change-me-admin-key",
    "company": "bookshop",
    "version": "1",
    "capability": "order.get",
    "secret_name": "orders_api_key",
    "server_id": "",
    "chat_id": "",
    "action_id": "",
    "run_id": "",
    "cursor": "",
}


def collection() -> dict[str, Any]:
    return {
        "info": {
            "name": "Support Guidance Assistant API (v2)",
            "description": (
                "Every endpoint of the Support Guidance Assistant. Set `agent_key` / "
                "`admin_key` (the values from AGENT_API_KEYS / ADMIN_API_KEYS, without the "
                "`label:` prefix) and `baseUrl` in the environment. Requests that create "
                "resources save their ids (`chat_id`, `action_id`, `server_id`, `run_id`, "
                "`version`, `cursor`) into collection variables for the next requests.\n\n"
                "Suggested order: Companies -> Guidelines/Upload -> Integrations (bind, test, "
                "go-live) -> Chats -> Actions -> Evals -> Internal.\n\n"
                "Generated by `scripts/export_postman.py`."
            ),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "item": ITEMS,
        "variable": [{"key": k, "value": v} for k, v in VARIABLES.items()],
    }


def environment() -> dict[str, Any]:
    keys = ("baseUrl", "agent_key", "admin_key", "company")
    return {
        "name": "Support Assistant — local",
        "values": [
            {
                "key": k,
                "value": VARIABLES[k],
                "type": "secret" if k.endswith("_key") else "default",
                "enabled": True,
            }
            for k in keys
        ],
        "_postman_variable_scope": "environment",
    }


def _normalise(path: str) -> str:
    """Map both Postman and OpenAPI paths to a comparable template."""
    path = path.replace("{{baseUrl}}", "").split("?")[0]
    path = re.sub(r"\{\{[^}]+\}\}", "{}", path)
    path = re.sub(r"\{[^}]+\}", "{}", path)
    # concrete capability names in binding paths count as the {capability} template
    return re.sub(r"(capability-bindings/)[a-z_]+\.[a-z_]+", r"\1{}", path)


def check_coverage(items: list[dict[str, Any]]) -> None:
    from app.main import create_app
    from tests.conftest import make_settings

    schema = create_app(make_settings()).openapi()
    expected = {
        (m.upper(), _normalise(p))
        for p, ops in schema["paths"].items()
        for m in ops
        if m in {"get", "post", "put", "patch", "delete"}
    }
    have = {
        (r["request"]["method"], _normalise(r["request"]["url"]["raw"]))
        for f in items
        for r in f["item"]
    }
    missing = expected - have
    unknown = have - expected
    if missing or unknown:
        raise SystemExit(f"Postman collection out of sync. missing={missing} unknown={unknown}")
    n_requests = sum(len(f["item"]) for f in items)
    print(f"Covers all {len(expected)} API operations ({n_requests} requests).")


def main() -> None:
    check_coverage(ITEMS)
    OUT.mkdir(exist_ok=True)
    (OUT / "support-assistant.postman_collection.json").write_text(
        json.dumps(collection(), indent=2), encoding="utf-8"
    )
    (OUT / "support-assistant.local.postman_environment.json").write_text(
        json.dumps(environment(), indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUT}/support-assistant.postman_collection.json and the local environment.")


if __name__ == "__main__":
    main()
