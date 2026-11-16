# Support Guidance Assistant — Architecture

This page shows the target architecture described in [claude.md](claude.md): diagrams first, then the key decisions and why they were made.

**Contents**
1. [System context](#1-system-context)
2. [Components and layering](#2-components-and-layering)
3. [Ingestion and compilation](#3-ingestion-and-compilation)
4. [Guideline version lifecycle](#4-guideline-version-lifecycle)
5. [Chat harness (LangGraph)](#5-chat-harness-langgraph)
6. [Chat request with tools and approval](#6-chat-request-with-tools-and-approval)
7. [Hybrid intent routing](#7-hybrid-intent-routing)
8. [Capabilities and bindings](#8-capabilities-and-bindings)
9. [Data model](#9-data-model)
10. [Deployment](#10-deployment)
11. [Key decisions](#11-key-decisions)

---

## 1. System context

```mermaid
flowchart LR
    agent["Support agent<br/>(agent API key)"]
    admin["Admin<br/>(admin API key)"]

    subgraph app["Support Guidance API (FastAPI)"]
        api["REST API /api/v1"]
    end

    pg[("Supabase Postgres<br/>+ pgvector")]
    gemini["Google Gemini<br/>generator / compiler / router<br/>+ embeddings"]

    subgraph company["Company systems (per tenant, optional)"]
        mcp["MCP server"]
        http["HTTP API"]
        db[("Company database<br/>read-only")]
    end

    prom["Prometheus"]
    ls["LangSmith<br/>(optional, off by default)"]
    redis[("Redis<br/>rate-limit counters<br/>(prod; memory:// locally)")]

    agent -- "chats, actions, feedback" --> api
    admin -- "companies, guidelines, integrations, evals, metrics" --> api
    api <--> pg
    api -- "LLM + embedding calls" --> gemini
    api -- "capability calls" --> mcp
    api -- "capability calls" --> http
    api -- "parameterized SELECT" --> db
    prom -- "scrape /metrics" --> api
    api -.-> ls
    api -- "POST /chats quota hits" --> redis
```

---

## 2. Components and layering

Dependencies only point downward. Routers hold no business logic or SQL, services never import FastAPI, and only the adapters know about MCP, HTTP or company databases.

```mermaid
flowchart TB
    subgraph API["api/v1 (routers)"]
        r1[companies] --- r2[guidelines] --- r3[chats] --- r4[actions] --- r5[integrations] --- r6[evals] --- r7[internal] --- r8[health]
    end

    subgraph SVC["services"]
        s1[company] --- s2[ingestion] --- s3[review] --- s4[guidance] --- s5[action] --- s6[integration] --- s7[eval] --- s8[metrics] --- s9[chat_history] --- s10["rate_limiter (limits)"]
    end

    subgraph HARNESS["harness (LangGraph)"]
        h1[graph] --- h2[nodes] --- h3[state]
    end

    subgraph AI["ai"]
        a1[providers] --- a2[compiler] --- a3[examples] --- a4[router] --- a5[context] --- a6[guidance_chain] --- a7[prompts] --- a8[schemas] --- a9["heuristic (offline)"]
    end

    subgraph CAP["capabilities"]
        c1[registry] --- c2[gateway] --- c3[policy] --- c4[secrets]
        c2 --> ad["adapters: mcp / http / sql_readonly / mock"]
    end

    subgraph DOCS["documents"]
        d1["parser registry: txt (pdf/docx later)"]
    end

    subgraph SEARCH["search"]
        se1["SearchGateway: pgvector + tsvector, RRF"]
    end

    subgraph REPO["repositories (all SQL)"]
        re["companies, versions, playbooks, search, chats, capabilities, evals, metrics"]
    end

    API --> SVC
    SVC --> HARNESS
    SVC --> DOCS
    SVC --> AI
    SVC --> CAP
    HARNESS --> AI
    HARNESS --> SEARCH
    HARNESS --> CAP
    HARNESS -- "unit of work" --> REPO
    SEARCH --> REPO
    SVC --> REPO
    CAP --> REPO
    REPO --> PG[("Postgres + pgvector")]
```

---

## 3. Ingestion and compilation

Ingestion is one pure async function with no FastAPI imports, so it can move to a worker queue unchanged. It is deliberately not a graph.

```mermaid
flowchart TD
    up["POST /companies/{slug}/guidelines<br/>file, activate=true, notes"] --> val{"Validate<br/>format / size / encoding / empty"}
    val -- "bad" --> e422["422 UNSUPPORTED_FORMAT / INVALID_FILE<br/>413 FILE_TOO_LARGE"]
    val -- "ok" --> lock["Lock company row<br/>(SELECT … FOR UPDATE)"]
    lock --> dup{"SHA-256 equals<br/>active version?"}
    dup -- "yes" --> e409["409 DUPLICATE_GUIDELINES"]
    dup -- "no" --> row["Create version row, commit<br/>version_number = max + 1<br/>status=processing, review=pending<br/>(retry once on IntegrityError)"]
    row --> ret["202 Accepted, pipeline in a background task<br/>(inline + 201 when INGESTION_SYNC)"]
    row --> parse["Parse into sections<br/>(DocumentParser: txt)"]
    parse --> comp["Compile with COMPILER_MODEL<br/>intents + steps + capabilities<br/>company rules, reference facts<br/>reuse previous active intent keys"]
    comp --> chk{"Validate<br/>section refs, capabilities,<br/>unique keys, coverage"}
    chk -- "blocking error" --> fail
    chk -- "ok / warnings" --> ex["Generate examples<br/>index + held-out eval + negatives"]
    ex --> emb["Embed sections, examples, facts"]
    emb --> tx["One transaction:<br/>insert compiled rows + synthetic eval cases<br/>status=ready"]
    tx --> ev["Retrieval-mode routing eval<br/>(no LLM)"]
    ev --> gate{"AUTO_APPROVE on<br/>and top-1 >= threshold?"}
    ev -. "eval error" .-> pend
    gate -- "yes" --> appr["review_status=approved<br/>(reviewed_by=auto)"]
    gate -- "no" --> pend["review_status=pending<br/>(admin approves later)"]
    appr --> act{"activate flag?"}
    act -- "yes" --> newer{"newer version<br/>already active?"}
    newer -- "no" --> live["Set companies.active_version_id"]
    newer -- "yes" --> idle
    act -- "no" --> idle["Ready, inactive"]
    live --> inv["Invalidate company cache"]
    idle --> inv
    pend --> inv
    parse -. "any exception<br/>parse … persist" .-> fail["status=failed, error_message<br/>no compiled rows kept<br/>previous active keeps serving"]
```

---

## 4. Guideline version lifecycle

```mermaid
stateDiagram-v2
    [*] --> Processing: upload
    Processing --> Failed: parse / compile / embed error
    Processing --> ReadyPending: compiled, eval below gate, eval failed or auto-approve off
    Processing --> ReadyApproved: compiled, auto-approved

    ReadyPending --> ReadyApproved: POST approve
    ReadyPending --> Rejected: POST reject
    ReadyApproved --> Rejected: POST reject (inactive only)
    Rejected --> ReadyApproved: POST approve

    ReadyApproved --> Active: activate (default on upload / approve, unless a newer version is active)
    Active --> ReadyApproved: another version activated
    ReadyApproved --> Active: POST activate (rollback / forward)

    Failed --> [*]: DELETE
    ReadyPending --> [*]: DELETE
    ReadyApproved --> [*]: DELETE
    Rejected --> [*]: DELETE

    note right of Active
        Exactly one per company.
        companies.active_version_id
        is the single source of truth.
        DELETE refused while active.
    end note
```

---

## 5. Chat harness (LangGraph)

The graph (`harness/graph.py`) is built once at startup and runs in-process. Each node is a plain async function, so it can be unit-tested without the graph. There is no checkpointer: one chat is one graph run. Before the graph runs, `GuidanceService` resolves the company context; a missing company (404) or no usable active version (409 `NO_ACTIVE_GUIDELINES`) is returned without running the graph.

```mermaid
flowchart TD
    start(["POST /chats"]) --> svc["GuidanceService.resolve_company<br/>company, active version, rules, live bindings<br/>(30s TTL cache)"]
    svc -- "404 / 409" --> rej(["error response, no graph run"])
    svc --> load["load_context<br/>embed message + notes once<br/>(version's embedding model)"]
    load --> route["route<br/>search_intents (ROUTING_CANDIDATES)<br/>-> ROUTER_MODEL picks up to ROUTING_MAX_INTENTS<br/>+ entities (regex + router)"]
    load --> facts["retrieve_facts<br/>hybrid search, REFERENCE_TOP_K"]
    route --> join["after_retrieval<br/>(join)"]
    facts --> join
    join -- "no intent selected<br/>(no_match)" --> fb["fallback_sections<br/>hybrid search, FALLBACK_SECTION_TOP_K"]
    join -- "intents selected" --> pre
    fb --> pre["prefetch<br/>live-bound lookup steps whose args<br/>are known (order id, email): no LLM"]
    pre --> agent["tool_agent<br/>tool model bound to the selected intents'<br/>bound READ capabilities only"]
    agent -- "AI message has tool_calls" --> tools["tools<br/>allowed reads via capability gateway<br/>(trigger=agent)"]
    tools --> agent
    agent -- "done: nothing left, no tool calls,<br/>AGENT_MAX_STEPS, deadline or model error" --> gen
    gen["generate<br/>GuidanceOutput (structured output)<br/>context lists live write capabilities"] --> gate["validate_and_gate<br/>drop unknown citations + intents<br/>schema + policy check proposals (proposed / denied)<br/>set performed_by in code<br/>no_match: low confidence + escalation"]
    gate --> persist["persist (always)<br/>chat + routing + references<br/>+ tool_calls + proposed_actions"]
    persist --> done(["201 success / 502 on error"])

    join -. "earlier error<br/>(load / route / facts)" .-> persist
    fb -. "error" .-> persist
    pre -. "error" .-> persist
    agent -. "error" .-> persist
    gen -. "error" .-> persist
```

Every node is wrapped by `guarded()`: an exception is recorded in `errors`, later nodes are skipped and the run ends in `persist`, so the chat row is stored with `status=error`. The service then maps router or generator failures to `502 LLM_ERROR` and search failures to `502 RETRIEVAL_ERROR`. Tool failures never fail a chat; the step just becomes manual.

`performed_by` on each plan step:

| Value | Meaning |
|---|---|
| `ai` | Say / ask steps folded into the drafted reply |
| `ai_verified` | Lookup the harness actually ran (links a `tool_call_id`) |
| `needs_approval` | Write proposed as an action (links a `proposed_action_id`) |
| `agent` | Manual: unbound, failed or timed-out lookups and actions, escalations |

---

## 6. Chat request with tools and approval

```mermaid
sequenceDiagram
    autonumber
    actor Agent as Support agent
    participant API as FastAPI + GuidanceService
    participant RL as Rate limiter (Redis / memory)
    participant G as Harness (LangGraph)
    participant E as Embeddings
    participant S as SearchGateway (pgvector)
    participant R as Router LLM
    participant CG as Capability gateway
    participant X as Company system (MCP/HTTP/SQL)
    participant L as Generator / tool LLM
    participant DB as Postgres

    Agent->>API: POST /chats {company, message, notes}
    API->>RL: check_chat(company slug, API-key label)
    alt company or agent quota exhausted
        RL-->>API: RateLimited
        API-->>Agent: 429 RATE_LIMITED + Retry-After (no chat row, no LLM call)
    end
    RL-->>API: ok + remaining (X-RateLimit-* headers)
    API->>DB: company + active version + rules + live bindings (30s cache)
    API->>G: ainvoke(state, recursion_limit)
    G->>E: load_context: embed message once
    par route
        G->>S: search_intents (vector + text, RRF, max per intent)
        S-->>G: top-20 candidates
        G->>R: select intents + extract entities
        R-->>G: [billing_issue, connection_failure], {email}
    and retrieve_facts
        G->>S: search_facts
        S-->>G: top-k facts
    end
    G->>CG: prefetch subscription.get(email)
    CG->>X: call with injected secret, timeout
    X-->>CG: result
    CG-->>G: audit record (redacted args, truncated result)
    opt lookups still missing args
        G->>L: tool_agent: bound read tools only
        L-->>G: tool_calls
        G->>CG: tools node (trigger=agent)
        CG-->>G: audit record
    end
    G->>L: generate: rules + playbooks + facts + tool results + message
    L-->>G: GuidanceOutput incl. proposed refund.create
    G->>G: validate_and_gate: citations, schema + policy in code, performed_by
    G->>DB: persist: chat + tool_calls + proposed_actions (status=proposed)
    G-->>API: final state
    API-->>Agent: 201 guidance + proposed actions

    Note over Agent,DB: Later, a separate request (no graph, no LLM)
    Agent->>API: POST /chats/{id}/actions/{aid}/approve
    API->>DB: lock action (FOR UPDATE), must be proposed
    API->>API: re-check live binding + policy (unbound = 409 CAPABILITY_NOT_BOUND, policy fail = denied + 409 POLICY_VIOLATION)
    API->>DB: status=executing, commit
    API->>CG: execute with idempotency key (trigger=execution)
    CG->>X: refund.create
    X-->>CG: ok
    CG-->>API: audit record
    API->>DB: action executed / failed + tool_calls row
    API-->>Agent: 200 action (approving again returns the same result)
```

### Rate limiting
`POST /chats` is the expensive endpoint (embeddings, two LLM calls, tool calls), so the router checks quotas with the [`limits`](https://limits.readthedocs.io) library (`services/rate_limiter.py`) **after** auth and body validation and **before** the service runs. The company slug is in the JSON body, so this can't be done in middleware.

| Level | Key | Default | Purpose |
|---|---|---|---|
| Per agent per company | `("chat", "agent", slug, API-key label)` | `CHAT_RATE_LIMIT_PER_AGENT=20/minute` | One agent or script can't flood a tenant |
| Per company | `("chat", "company", slug)` | `CHAT_RATE_LIMIT_PER_COMPANY=300/minute` | Caps one tenant's total LLM spend |

- The company quota is tested first, then the agent quota is hit, then the company quota. A request rejected at one level doesn't use up quota at the other.
- Strategy: sliding-window counter by default (`moving-window` and `fixed-window` can be configured). Several windows per level are allowed (`20/minute;300/hour`).
- Rejected requests get `429 RATE_LIMITED`, `Retry-After` and `details.scope`. They aren't stored as chats but count in `chat_rate_limited_total{company,scope}`.
- Storage: `memory://` per process (local, tests), Redis in production so all replicas share counters.
- If Redis is unreachable, chats are allowed (`RATE_LIMIT_FAIL_OPEN=true`) and a warning is logged.
- An agent is identified by the label of their API key, so agents sharing a key share one quota.

---

## 7. Hybrid intent routing

The router never sees the full intent list, only about 20 candidates. That is what lets this scale to hundreds of intents per company.

```mermaid
flowchart LR
    msg["customer message<br/>+ agent notes"] --> emb["embed once in load_context<br/>(version's model + dims)"]
    msg --> tsq["OR tsquery built in code<br/>word | word | …<br/>(stop words removed)"]

    subgraph pg["Postgres, filtered by company_id + version_id"]
        vec["HNSW cosine on<br/>intent_examples.embedding<br/>split = index<br/>(ef_search, iterative_scan)"]
        txt["GIN ts_rank on<br/>intent_examples.tsv"]
    end

    emb --> vec
    tsq --> txt
    vec --> rrf["RRF fusion<br/>1 / (60 + rank)"]
    txt --> rrf
    rrf --> agg["max score per intent<br/>top ROUTING_CANDIDATES"]
    agg --> sel["Router LLM<br/>select <= ROUTING_MAX_INTENTS<br/>or no_match"]
    sel --> san["drop keys that weren't candidates<br/>de-duplicate, cap"]
    san --> out["selected intents<br/>+ entities"]
```

Facts and fallback sections use the same two arms and RRF, without the per-intent step. The offline router and retrieval evals use a retrieval-only no-match rule instead of the LLM: no candidates, or no keyword hit and best cosine similarity below `ROUTING_MIN_SCORE`.

---

## 8. Capabilities and bindings

Playbook steps refer to abstract capabilities. Each company decides how, or whether, each one is implemented.

```mermaid
flowchart LR
    step["playbook step<br/>type=lookup<br/>capability=order.get"] --> reg["capability registry<br/>name, kind, schemas, policy keys"]
    reg --> bind{"company binding<br/>status = live?"}
    bind -- "no" --> manual["manual step for agent"]
    bind -- "yes" --> gw["gateway:<br/>validate args, policy (writes),<br/>tenant from request + secret getter,<br/>TOOL_TIMEOUT_SECONDS,<br/>truncate + redact, audit record"]
    gw --> mcpA["mcp adapter<br/>mapped tool only<br/>SSRF guard"]
    gw --> httpA["http adapter<br/>templated request<br/>SSRF guard"]
    gw --> sqlA["sql_readonly adapter<br/>admin SELECT, READ ONLY tx"]
    gw --> mockA["mock adapter<br/>demo / tests"]
```

```mermaid
stateDiagram-v2
    [*] --> Draft: PUT binding
    Draft --> Tested: POST test (dry run ok)
    Tested --> Draft: PUT (config updated)
    Tested --> Live: POST go-live
    Live --> Draft: PUT (config updated)
    Draft --> [*]: DELETE
    Tested --> [*]: DELETE
    Live --> [*]: DELETE
    note right of Live
        Only live bindings are used
        by chats and action execution.
        go-live from draft: 409 BINDING_NOT_TESTED.
    end note
```

Proposed action lifecycle:

```mermaid
stateDiagram-v2
    [*] --> Denied: bad args or policy check fails at proposal
    [*] --> Proposed: args + policy ok
    Proposed --> Rejected: agent rejects
    Proposed --> Denied: policy re-check fails at approval
    Proposed --> Executing: agent approves (row locked)
    Executing --> Executed: adapter ok
    Executing --> Failed: adapter error / timeout
    Denied --> [*]
    Rejected --> [*]
    Executed --> [*]
    Failed --> [*]
```

Proposals for unknown, read-only or unbound capabilities are not stored; the step becomes manual (`performed_by: "agent"`).

---

## 9. Data model

```mermaid
erDiagram
    companies ||--o{ guideline_versions : has
    companies |o--o| guideline_versions : "active_version_id"
    guideline_versions ||--o{ guideline_sections : "parsed into"
    guideline_versions ||--o{ intents : "compiled into"
    intents ||--o{ intent_examples : "indexed by"
    guideline_versions ||--|| company_rules : "has one"
    guideline_versions ||--o{ reference_facts : has
    guideline_versions ||--o{ routing_eval_runs : "evaluated by"
    companies ||--o{ routing_eval_cases : owns
    chats |o--o{ routing_eval_cases : "feedback"
    companies ||--o{ chats : has
    guideline_versions |o--o{ chats : "used by"
    chats ||--o{ tool_calls : made
    chats ||--o{ proposed_actions : proposed
    proposed_actions ||--o{ tool_calls : "executed via"
    companies ||--o{ capability_bindings : configures
    companies ||--o{ mcp_servers : registers
    companies ||--o{ company_secrets : stores

    companies {
        uuid id PK
        text slug UK
        uuid active_version_id FK
    }
    guideline_versions {
        uuid id PK
        uuid company_id FK
        int version_number
        text status
        text review_status
        bool activate_on_approval
        text embedding_model
        int embedding_dimensions
        json coverage
    }
    guideline_sections {
        uuid id PK
        uuid version_id FK
        text heading
        text text
        vector embedding
        tsvector tsv
    }
    intents {
        uuid id PK
        uuid version_id FK
        text key
        json steps
        text reply_template
    }
    intent_examples {
        uuid id PK
        uuid intent_id FK
        text split
        vector embedding
        tsvector tsv
    }
    company_rules {
        uuid id PK
        uuid version_id FK
        json rules
    }
    reference_facts {
        uuid id PK
        uuid version_id FK
        text text
        vector embedding
        tsvector tsv
    }
    chats {
        uuid id PK
        uuid company_id FK
        uuid guideline_version_id FK
        text status
        json guidance
        json routing
        json references
    }
    capability_bindings {
        uuid id PK
        uuid company_id FK
        text capability
        text adapter_type
        json config
        json policy
        text status
    }
    tool_calls {
        uuid id PK
        uuid chat_id FK
        text capability
        text trigger
        text status
    }
    proposed_actions {
        uuid id PK
        uuid chat_id FK
        text capability
        text status
        text idempotency_key UK
    }
    routing_eval_cases {
        uuid id PK
        uuid company_id FK
        json expected_intent_keys
        text source
    }
    routing_eval_runs {
        uuid id PK
        uuid company_id FK
        uuid version_id FK
        text mode
        json metrics
    }
    mcp_servers {
        uuid id PK
        uuid company_id FK
        text url
    }
    company_secrets {
        uuid id PK
        uuid company_id FK
        text ciphertext
    }
```

All playbook tables carry `company_id` + `version_id` and cascade on version delete, so deleting a version also deletes its vectors.

---

## 10. Deployment

```mermaid
flowchart LR
    subgraph local["Local development"]
        dev["uvicorn (make run)<br/>or docker compose api"] --> sb[("Supabase CLI stack<br/>Postgres 17 + pgvector<br/>:55422")]
        tests["pytest integration"] --> sbt[("support_assistant_test<br/>same instance")]
    end

    subgraph prod["Production"]
        lb["load balancer"] --> a1["API replica"]
        lb --> a2["API replica"]
        a1 --> hs[("Hosted Supabase<br/>session pooler / direct :5432")]
        a2 --> hs
        a1 --> rd[("Redis<br/>rate-limit counters")]
        a2 --> rd
    end
```

The API process is stateless (all durable state lives in Postgres, shared rate-limit counters in Redis), so it scales horizontally. Locally the limiter uses in-process memory and needs no Redis. Only env vars differ between environments.

---

## 11. Key decisions

| # | Decision | Why | Alternatives considered | Trade-off |
|---|---|---|---|---|
| 1 | **Compile guidelines into structured playbooks at upload** | Guidelines are runbooks for humans. Turning them into intents, typed steps, rules and facts makes routing exact, makes steps citable, and links steps to tools. | Plain RAG over chunks | Needs an LLM compile step plus validation; mistakes there need a coverage check and an optional human review |
| 2 | **pgvector in the same Postgres; Chroma removed** | One store, one transaction, one backup. Tenant and version filters in SQL; cascade deletes. Hundreds of intents per company is small for pgvector. | Chroma, a dedicated vector DB | At very large scale, pgvector needs partitioning; `SearchGateway` keeps a swap possible |
| 3 | **Hybrid search (vector + full-text, RRF)** | Embeddings miss product names, codes and exact phrasing; keyword search catches them | Vector only | Two queries per search; both indexed and cheap |
| 4 | **Embed synthetic example messages per intent; score by best match** | Customer wording rarely matches an intent's name or description | Embedding the intent description | More rows and embedding cost at ingest |
| 5 | **Two-stage routing: search candidates, then a cheap LLM selects** | Scales to hundreds of intents (the router sees about 20); handles messages with several intents and no-match | Single classifier over all intents; vector top-1 | One extra (fast) LLM call per chat |
| 6 | **Company rules always injected, never retrieved** | Tone, sign-off and prohibitions apply to every reply; retrieval could drop them | Retrieve as chunks | Slightly larger prompt |
| 7 | **Reference facts as a separate retrieval index** | Prices, device lists and URLs aren't procedures and don't map to steps | Folding facts into intents | A second search per chat (runs in parallel) |
| 8 | **Stable intent keys across versions** | Metrics, evals, feedback and history stay comparable after re-uploads | New keys per version | Compiler must be given previous keys; covered by a test |
| 9 | **Immutable versions; single active pointer on `companies`** | Exactly one active version, instant rollback with no re-indexing | `is_active` flag on versions | Activation needs a row lock (`FOR UPDATE`) |
| 10 | **Auto-approve by default, gated by a retrieval eval** | Zero friction for normal uploads, but a poor compile doesn't go live silently | Always manual review; always auto | Gate threshold needs tuning per domain |
| 11 | **Routing evals keyed by intent key, run per version** | Lets you compare two guideline versions before a rollback; agent feedback becomes test data | Ad-hoc manual testing | Synthetic cases can be easier than real traffic, which feedback cases offset |
| 12 | **Abstract capabilities + per-company bindings (MCP / HTTP / read-only SQL / mock)** | Guidelines stay vendor-neutral; companies go live with zero integrations and add automation one binding at a time | Hard-coded vendor APIs; giving the model a whole MCP server | Admins must map tools to capabilities and test each binding |
| 13 | **Reads automatic, writes only as approved proposals** | Side effects stay under human control; the harness executes once with an idempotency key | Letting the model execute writes | One extra click for the agent |
| 14 | **Policy enforced in code, at proposal and at execution** | Prompts can be bypassed; code can't | Policy in the prompt | Policy keys must exist in the capability registry |
| 15 | **Read-only SQL only via admin-defined, parameterized SELECTs** | Reaches company databases without model-written SQL or write risk | Text-to-SQL | Each query must be defined up front |
| 16 | **Graceful degradation everywhere** | A broken or missing integration turns a step into a manual step instead of failing the chat; only an LLM failure returns 502 | Failing the request | Guidance may be less automated, never unavailable |
| 17 | **LangGraph for the chat harness, run in-process** | Branching, a bounded tool loop, parallel nodes and per-node tracing in one place; no extra service | Hand-written orchestration; LangGraph Platform | New dependency; nodes kept as plain functions to limit coupling |
| 18 | **No LangGraph checkpointer; approval is a separate deterministic request** | Paused graph state would have to live for minutes in storage; executing an approved action needs no LLM | `interrupt()` + Postgres checkpointer | The graph can't resume a conversation (multi-turn is out of scope) |
| 19 | **Ingestion is a plain async function, not a graph** | It's linear; plain code is simpler and moves to a worker queue unchanged | LangGraph for ingestion | None significant |
| 20 | **Alembic owns the schema; Supabase only hosts it** | One migration history for local, CI and prod; CI can use plain pgvector Postgres | Supabase migrations / Studio | Supabase-specific features aren't used |
| 21 | **Secrets encrypted (Fernet) behind `SecretStore`; injected at call time** | Credentials never reach prompts, logs or the audit table | Plaintext config | Key management via env var until KMS/Vault |
| 22 | **Everything external is untrusted data** | Customer messages, guidelines and tool output are all delimited and can't change instructions (prompt-injection defence) | — | Prompt is a bit more verbose |
| 23 | **LangSmith tracing off by default** | Tracing would send customer messages and tool data to a third party | Always on | Opt-in through env vars when debugging |
| 24 | **Chat rate limits per agent per company and per company, using `limits` in the router; Redis in production** | The company slug is in the body, so the check runs after parsing and before any LLM spend. `limits` has async storage, several algorithms and swappable backends (memory locally, Redis shared across replicas). | slowapi (key function only sees the request, not the body); fastapi-limiter (Redis only); gateway/edge limits (can't see company or agent); a Postgres counter table | Adds Redis in production. Fails open on Redis errors, so limits can lapse during an outage. "Agent" means API-key label. |

### Out of scope (future work)
Multi-turn threads, SSO/RBAC, a queue worker, multilingual messages, PDF/DOCX/OCR parsers (the interface exists), native OAuth connectors (Shopify, Stripe, Zendesk) and OpenAPI import, KMS/Vault, per-company tool rate limits, cross-encoder re-ranking, LLM-judge reply evals, partitioning by company, streaming responses.
