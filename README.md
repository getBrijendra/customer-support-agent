# Support Guidance Assistant

A FastAPI back end that gives customer-support agents an AI-recommended reply plus handling
instructions for a customer message. The guidance is grounded in **that company's own support
guidelines**, which are retrieved with RAG from versioned `.txt` uploads.

- **LLM and embeddings:** Google Gemini via LangChain. Model names come from config; the defaults are `gemini-3.1-pro-preview` and `gemini-embedding-2`.
- **Relational DB:** Supabase Postgres. Local development uses the Supabase CLI and production uses hosted Supabase. The app uses SQLAlchemy 2 async with asyncpg, and **Alembic owns the schema**.
- **Vectors:** ChromaDB in local persistent mode, behind a `VectorStoreGateway` interface. Each (company, version) pair gets its own collection.

```
api (routers, no logic/SQL) ─▶ services ─▶ repositories (all SQL)
                                        ├▶ vectorstore (only place importing chromadb)
                                        └▶ ai (providers, prompt, structured-output chain)
```

---

## First-time local setup

1. Install Docker and the Supabase CLI (`brew install supabase/tap/supabase`, the official binary, or use `npx supabase`; the Makefile falls back to `npx` automatically). Then run `uv sync`.
2. Run `cp .env.example .env`, then add `GOOGLE_API_KEY` and your own `AGENT_API_KEYS` / `ADMIN_API_KEYS`.
3. Run `make db-up`, then confirm the URL with `supabase status`.
4. Run `make migrate && make seed`.
5. Run `make run` and open <http://127.0.0.1:8000/docs>.
6. Run `make test`.

> **Ports:** `supabase/config.toml` moves this project to the **554xx** port range (DB `55422`,
> Studio `55423`, API `55421`) so it can run alongside other local Supabase projects on the
> default 543xx ports. Always copy the DB URL from `supabase status` into `.env`. Change the
> scheme to `postgresql+asyncpg://`.

`make db-up` runs `supabase start` and then `scripts/create_test_db.py`. That script creates
the `support_assistant_test` database for the integration tests, using asyncpg so `psql` isn't
needed. `make db-down` stops the stack; your data survives restarts. `make db-reset` wipes the
local DB and re-runs the migrations and the seed.

### Seed data
`scripts/seed.py` creates `bookshop` ("Bookshop") and `vpn-saas` ("VPN SaaS"). It then uploads
`data/seed_guidelines/bookshop.txt` and `data/seed_guidelines/vpn_saas.txt` through the service
layer (synchronously), and each upload becomes v1 and active. **These two files must be added
by you.** The script stops with a clear message if they are missing. Re-running the script is
safe: existing companies are reused, and a file identical to the active version is skipped.

### Windows note
If `uv run` warns that `VIRTUAL_ENV ... does not match the project environment`, an unrelated
virtualenv is active. Run `deactivate` first, or prefix commands with `VIRTUAL_ENV=`.

---

## Configuration

`.env.example` documents every variable; the main ones are listed below. `.env` is git-ignored.

| Variable | Notes |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` Local: Supabase CLI DB. Prod: hosted Supabase. |
| `TEST_DATABASE_URL` | Separate DB for the integration tests. Its name **must end in `_test`**. |
| `GOOGLE_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL` | Gemini settings. Without a key the app still starts: chats return `502 LLM_ERROR` and uploads end `failed`. |
| `VECTOR_BACKEND` | `chroma_local` (default) or `chroma_remote` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `RAG_TOP_K` / `RAG_SEARCH_TYPE` | RAG tuning (`mmr` or `similarity`) |
| `INGESTION_SYNC` | `true` ingests inline and returns 201. `false` returns 202 and ingests in the background. |
| `AGENT_API_KEYS` / `ADMIN_API_KEYS` | Comma-separated keys. Use `label:key` to record a readable `uploaded_by` label. |

### Supabase connection pooler
Hosted Supabase offers three connection types: a direct connection, a **session pooler (port
5432)** and a **transaction pooler (port 6543)**. Use the direct connection or the session
pooler for the API process.

If `DATABASE_URL` uses port **6543**, asyncpg prepared statements break under PgBouncer
transaction mode. The app detects this case and creates the engine with
`statement_cache_size=0`, `prepared_statement_cache_size=0` and `NullPool`. Alembic does the
same. It works, but you lose statement caching and app-side pooling.

### Schema ownership: Alembic only
Alembic (`migrations/`) is the **single source of truth** for the schema, both locally and in
production. Do **not** add tables in `supabase/migrations/` or create them in Studio, because
that would create two competing migration histories. `supabase/` contains only `config.toml`.
If you ever run `supabase db reset`, re-run `alembic upgrade head` and the seed (`make db-reset`
does both). The integration suite runs `alembic upgrade head` on a freshly created database.
It also runs an upgrade → downgrade base → upgrade cycle, so every migration is tested.

### Production
1. Copy `.env.prod.example`. Set `DATABASE_URL` to the hosted Supabase connection string (direct connection or session pooler on port 5432), plus your secrets.
2. Run `alembic upgrade head` against the hosted database as a release step (from CI/CD or manually) before you deploy.
3. Deploy the image from the `Dockerfile`. Mount a persistent volume at `CHROMA_PERSIST_DIR` for as long as you use local Chroma.

Production differs from local only in `DATABASE_URL` and secrets. The hosted project's Postgres
major version should match `db.major_version` in `supabase/config.toml` (currently **17**) and
the `postgres:17` service used in CI. Check the version in the Supabase dashboard under
**Settings → Infrastructure** and keep all three in sync.

### Docker (optional)
`docker compose up --build` runs only the API container. The database still comes from the
Supabase CLI on the host. Set `DATABASE_URL=postgresql+asyncpg://postgres:postgres@host.docker.internal:55422/postgres`.
The compose file adds `extra_hosts: host.docker.internal:host-gateway` for Linux, and Chroma
data lives in the `chroma_data` volume.

---

## How it works

### Guideline versions
- Every upload creates a new, **immutable** version (`v1`, `v2`, …) with its own Chroma collection, named `gl_{company_id.hex[:12]}_v{n}`. The raw text is kept in Postgres, so vectors can always be rebuilt.
- **Version numbers** are allocated inside a transaction that first locks the company row (`SELECT … FOR UPDATE`). The UNIQUE(company_id, version_number) constraint acts as a backstop, with one retry on conflict.
- **Exactly one active version.** The active version is a single pointer, `companies.active_version_id`. Activation (rollback or roll-forward) just moves the pointer, so no re-embedding is needed. With `activate=true` (the default), a newly ready version becomes active unless an even newer version is already active. That rule prevents a slow, older concurrent upload from overriding a newer one.
- **Failed ingestion** marks the version `failed` and drops its partial collection. The previous active version keeps serving.
- **Duplicates.** Uploading the exact content of the *active* version returns `409 DUPLICATE_GUIDELINES`. Content identical to an older inactive version is accepted, and the response includes an `info` hint suggesting you activate that version instead.
- Ingestion (`IngestionService.run_ingestion`) has no FastAPI dependency, so it can move to a worker queue unchanged.

### Chunking
`RecursiveCharacterTextSplitter` is configured to split before markdown, numbered and ALL-CAPS
headings first, then at blank lines, lines, sentences and words. Each chunk gets:
- the nearest heading as its `section`
- a deterministic id, `{version_id}:{chunk_index:04d}`, so re-ingesting a version upserts the same chunks and doesn't duplicate them

### Chats
1. Resolve the company and its active version. This uses a 30s in-process cache, which is invalidated on upload, activation and deletion.
2. Retrieve the top-k chunks from that version's collection, using the **embedding model recorded on the version**. MMR is computed locally, so every chunk keeps a cosine score.
3. Run `prompt | llm.with_structured_output(GuidanceOutput, include_raw=True)` and read token usage from the raw `AIMessage`.
4. **Persist every attempt**, including failures, with latencies (total, retrieval, LLM), tokens, model, prompt version, retrieved chunks and request id.

Guideline references are filtered to chunk ids that were actually retrieved, so citations can't be hallucinated.

The prompt (`app/ai/prompts.py`, `PROMPT_VERSION = "v1"`) does the following:
- Treats the customer message, the agent notes and the guideline text as **data, not instructions**.
- Forbids inventing policies.
- Requires `confidence="low"` plus escalation or `missing_information` when the guidelines don't cover the case.

### Observability
- **JSON logs.** Every line carries `request_id`. Each request also gets one access line with `path`, `method`, `status` and `duration_ms`.
- **Request ids.** `X-Request-ID` is accepted on the way in, or generated, and always returned.
- **Domain events:** `chat.completed`, `chat.failed`, `ingestion.started`, `ingestion.completed`, `ingestion.failed`, `guidelines.activated`.
- **No message content in logs.** Customer messages are logged only as length plus a short hash, and API keys are never logged.
- **Prometheus** (`/metrics`, admin only): HTTP metrics plus `chat_requests_total{company,status}`, `chat_latency_seconds`, `llm_tokens_total{direction}` and `ingestion_total{status}`.
- **Metrics summary** (`/api/v1/internal/metrics/summary`) is computed in Postgres: `percentile_cont` for p50/p95, `date_trunc` for daily volume, and JSONB for escalation and confidence. The `/api/v1/internal/dashboard` page renders it with Chart.js.

---

## API (all under `/api/v1`; auth header `X-API-Key`)

**Agent** keys can create chats and read companies and guidelines. **Admin** keys can do
everything. The health endpoints are public. Every error has the shape
`{"error": {"code", "message", "request_id"}}`.

```bash
API=http://127.0.0.1:8000
AGENT="X-API-Key: <agent-key>"
ADMIN="X-API-Key: <admin-key>"

# Health
curl $API/api/v1/health
curl $API/api/v1/ready                       # DB SELECT 1 + vector store heartbeat

# Companies
curl -X POST $API/api/v1/companies -H "$ADMIN" -H 'Content-Type: application/json' \
     -d '{"slug":"bookshop","name":"Bookshop","description":"Online bookshop"}'
curl $API/api/v1/companies -H "$AGENT"
curl $API/api/v1/companies/bookshop -H "$AGENT"

# Guideline versions
curl -X POST $API/api/v1/companies/bookshop/guidelines -H "$ADMIN" \
     -F file=@data/seed_guidelines/bookshop.txt -F activate=true -F notes="refund policy updated"
curl $API/api/v1/companies/bookshop/guidelines -H "$AGENT"                 # newest first, is_active flag
curl $API/api/v1/companies/bookshop/guidelines/active -H "$AGENT"
curl "$API/api/v1/companies/bookshop/guidelines/2?include_text=true" -H "$AGENT"
curl -X POST $API/api/v1/companies/bookshop/guidelines/1/activate -H "$ADMIN"   # rollback
curl -X DELETE $API/api/v1/companies/bookshop/guidelines/3 -H "$ADMIN"          # not the active one

# Chats
curl -X POST $API/api/v1/chats -H "$AGENT" -H 'Content-Type: application/json' -d '{
  "company": "vpn-saas",
  "customer_message": "I was charged twice this month and the VPN will not connect on my iPhone.",
  "agent_notes": "Customer is on annual plan",
  "channel": "email"}'
curl "$API/api/v1/chats?company=vpn-saas&status=success&from=2026-11-01T00:00:00%2B01:00&limit=20" -H "$AGENT"
curl "$API/api/v1/chats?cursor=<next_cursor>" -H "$AGENT"
curl $API/api/v1/chats/<chat_id> -H "$AGENT"

# Internal (admin)
curl "$API/api/v1/internal/metrics/summary?from=2026-11-01T00:00:00Z&to=2026-11-10T00:00:00Z&company=bookshop" -H "$ADMIN"
open "$API/api/v1/internal/dashboard?key=<admin-key>"    # or send the X-API-Key header
curl $API/metrics -H "$ADMIN"
```

Error codes: `COMPANY_NOT_FOUND`, `COMPANY_EXISTS`, `VERSION_NOT_FOUND`, `NO_ACTIVE_GUIDELINES`,
`VERSION_NOT_READY`, `VERSION_IS_ACTIVE`, `DUPLICATE_GUIDELINES`, `INVALID_FILE`,
`FILE_TOO_LARGE`, `CHAT_NOT_FOUND`, `UNAUTHORIZED`, `FORBIDDEN`, `LLM_ERROR`,
`RETRIEVAL_ERROR`, `VALIDATION_ERROR`, `INTERNAL_ERROR`.

---

## Switching to Chroma Cloud or a remote Chroma server

Only configuration changes. Nothing outside `app/vectorstore/` imports `chromadb`.

```env
VECTOR_BACKEND=chroma_remote
# Chroma Cloud:
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
# or a self-hosted server:
CHROMA_HOST=chroma.internal
CHROMA_PORT=8000
CHROMA_API_KEY=...        # optional token auth
```

Existing local collections are not migrated automatically. To move them, re-run ingestion
for the versions you need: the raw text is in Postgres and chunk ids are deterministic. With a
remote store the API becomes fully stateless and can scale horizontally.

---

## Tests

| Command | What it runs |
|---|---|
| `make test-unit` | `pytest -m unit`: about 60 fast tests, no Docker or Postgres. Only the company-repository CRUD test uses in-memory SQLite. |
| `make test-integration` | `pytest -m integration`: the real app against real Postgres (`TEST_DATABASE_URL`). |
| `make test` | Both tiers. Plain `pytest` does the same and skips the integration tier with a hint if Postgres is unreachable. |
| `make test-live` | Opt-in smoke test with real Gemini (needs `GOOGLE_API_KEY` and `bookshop.txt`). |

How the tiers work:
- No test needs a Gemini key: they use `DeterministicFakeEmbedding` and a fake generator that implements the same interface as the real chain.
- At session start, the integration harness **drops and recreates** the test database and runs `alembic upgrade head`. Between tests it runs `TRUNCATE … RESTART IDENTITY CASCADE`.
- The harness refuses to run unless the database name ends in `_test` and differs from `DATABASE_URL`.

CI (`.github/workflows/ci.yml`) has two jobs:
1. ruff, `ruff format --check`, mypy and the unit tier.
2. The integration tier against a `postgres:17` service container. A plain Postgres image is fine because Alembic owns the schema and the app uses only standard Postgres features.

Lint and type checks: `make lint`, `make typecheck`.

---

## Future work (out of scope)

- Multi-turn conversation threads, user accounts/SSO and per-company RBAC.
- A real queue worker for ingestion (Arq/Celery). The ingestion function is already queue-ready.
- Re-ranking with a cross-encoder, and hybrid keyword + vector search.
- An evaluation harness for answer quality.
- PDF/DOCX guideline formats and streaming responses.
- Rebuilding collections automatically when switching vector backends or embedding models.
