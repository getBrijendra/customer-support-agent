# Support Guidance Assistant — Architecture (v2)

This page shows the target architecture described in [claude.md](claude.md) (the v2 spec): diagrams first, then the key decisions and why they were made.

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

    agent -- "chats, actions, feedback" --> api
    admin -- "companies, guidelines, integrations, evals, metrics" --> api
    api <--> pg
    api -- "LLM + embedding calls" --> gemini
    api -- "capability calls" --> mcp
    api -- "capability calls" --> http
    api -- "parameterized SELECT" --> db
    prom -- "scrape /metrics" --> api
    api -.-> ls
```

---

## 2. Components and layering

Dependencies only point downward. Routers hold no business logic or SQL, services never import FastAPI, and only the adapters know about MCP, HTTP or company databases.

```mermaid
flowchart TB
    subgraph API["api/v1 (routers)"]
        r1[companies] --- r2[guidelines] --- r3[chats] --- r4[actions] --- r5[integrations] --- r6[evals] --- r7[internal]
    end

    subgraph SVC["services"]
        s1[company] --- s2[ingestion] --- s3[review] --- s4[guidance] --- s5[action] --- s6[integration] --- s7[eval] --- s8[metrics]
    end

    subgraph HARNESS["harness (LangGraph)"]
        h1[graph] --- h2[nodes] --- h3[state]
    end

    subgraph AI["ai"]
        a1[providers] --- a2[compiler] --- a3[router] --- a4[context] --- a5[prompts] --- a6[schemas]
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
    up["POST /companies/{slug}/guidelines<br/>file, activate=true, notes"] --> val{"Validate<br/>format / size / encoding"}
    val -- "bad" --> e422["422 / 413"]
    val -- "ok" --> dup{"SHA-256 equals<br/>active version?"}
    dup -- "yes" --> e409["409 DUPLICATE_GUIDELINES"]
    dup -- "no" --> row["Create version row<br/>status=processing, review=pending<br/>FOR UPDATE on company"]
    row --> ret["202 Accepted<br/>(201 when INGESTION_SYNC)"]
    row --> parse["Parse into sections<br/>(DocumentParser: txt)"]
    parse --> comp["Compile with COMPILER_MODEL<br/>intents + steps + capabilities<br/>company rules, reference facts<br/>reuse previous intent keys"]
    comp --> chk{"Validate<br/>section refs, capabilities,<br/>unique keys, coverage"}
    chk -- "blocking error" --> fail
    chk -- "ok / warnings" --> ex["Generate examples<br/>index + held-out eval + negatives"]
    ex --> emb["Embed sections, examples, facts"]
    emb --> tx["One transaction:<br/>insert compiled rows, status=ready"]
    tx --> ev["Retrieval-mode routing eval"]
    ev --> gate{"AUTO_APPROVE on<br/>and top-1 >= threshold?"}
    gate -- "yes" --> appr["review_status=approved"]
    gate -- "no" --> pend["review_status=pending<br/>(admin approves later)"]
    appr --> act{"activate flag?"}
    act -- "yes" --> live["Set companies.active_version_id<br/>invalidate cache"]
    act -- "no" --> idle["Ready, inactive"]
    parse -. "exception" .-> fail["status=failed, rollback<br/>previous active keeps serving"]
    comp -. "exception" .-> fail
    emb -. "exception" .-> fail
```

---

## 4. Guideline version lifecycle

```mermaid
stateDiagram-v2
    [*] --> Processing: upload
    Processing --> Failed: parse / compile / embed error
    Processing --> ReadyPending: compiled, eval below gate or auto-approve off
    Processing --> ReadyApproved: compiled, auto-approved

    ReadyPending --> ReadyApproved: POST approve
    ReadyPending --> Rejected: POST reject
    Rejected --> ReadyApproved: POST approve

    ReadyApproved --> Active: activate (default on upload / approve)
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

The graph is built once at startup and runs in-process. Each node is a plain async function, so it can be unit-tested without the graph. There is no checkpointer: one chat is one graph run.

```mermaid
flowchart TD
    start(["POST /chats"]) --> load["load_context<br/>company + active version (TTL cache)<br/>embed message once"]
    load --> route["route<br/>hybrid candidates -> ROUTER_MODEL<br/>picks 0..3 intents + entities"]
    load --> facts["retrieve_facts<br/>hybrid search on reference facts"]
    route --> nm{"no_match or<br/>score < min?"}
    facts --> nm
    nm -- "yes" --> fb["fallback_sections<br/>section search, force low confidence<br/>+ escalation"]
    nm -- "no" --> pre
    fb --> pre["prefetch<br/>deterministic lookups when<br/>args are known (order id, email)"]
    pre --> ta{"lookups left?"}
    ta -- "yes" --> agent["tool_agent<br/>LLM bound to bound READ<br/>capabilities only"]
    agent --> tools["tools<br/>via capability gateway"]
    tools --> agent
    agent -- "done or limit hit<br/>(steps / timeouts)" --> gen
    ta -- "no" --> gen["generate<br/>GuidanceOutput (structured)"]
    gen --> gate["validate_and_gate<br/>strip unknown citations<br/>policy-check proposed actions<br/>tag performed_by"]
    gate --> persist["persist<br/>chat + routing + tool_calls<br/>+ proposed_actions"]
    persist --> done(["201 response"])

    load -. "error" .-> persist
    route -. "error" .-> persist
    gen -. "LLM error -> 502" .-> persist
```

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
    participant API as FastAPI
    participant G as Harness (LangGraph)
    participant S as SearchGateway (pgvector)
    participant R as Router LLM
    participant CG as Capability gateway
    participant X as Company system (MCP/HTTP/SQL)
    participant L as Generator LLM
    participant DB as Postgres

    Agent->>API: POST /chats {company, message, notes}
    API->>G: invoke(state)
    G->>DB: active version + rules (cached)
    par routing
        G->>S: search_intents (vector + text, RRF)
        S-->>G: top-20 candidates
        G->>R: select intents + extract entities
        R-->>G: [billing_issue, connection_failure], {email}
    and facts
        G->>S: search_facts
        S-->>G: top-k facts
    end
    G->>CG: prefetch subscription.get(email)
    CG->>X: call with injected secret, timeout
    X-->>CG: result
    CG->>DB: tool_calls row (redacted)
    CG-->>G: tool_result (untrusted, truncated)
    G->>L: rules + playbooks + facts + tool results + message
    L-->>G: GuidanceOutput incl. proposed refund.create
    G->>G: validate citations, policy check in code
    G->>DB: chat + proposed_actions (status=proposed)
    G-->>API: final state
    API-->>Agent: 201 guidance + proposed actions

    Note over Agent,DB: Later, a separate request (no graph, no LLM)
    Agent->>API: POST /chats/{id}/actions/{aid}/approve
    API->>DB: lock action (FOR UPDATE), status=executing
    API->>CG: re-check policy, execute with idempotency key
    CG->>X: refund.create
    X-->>CG: ok
    CG->>DB: tool_calls row, action=executed
    API-->>Agent: 200 executed (repeat approve won't run again)
```

---

## 7. Hybrid intent routing

The router never sees the full intent list, only about 20 candidates. That is what lets this scale to hundreds of intents per company.

```mermaid
flowchart LR
    msg["customer message<br/>+ agent notes"] --> emb["embed once<br/>(version's model + dims)"]
    msg --> tsq["websearch_to_tsquery"]

    subgraph pg["Postgres, filtered by company_id + version_id"]
        vec["HNSW cosine on<br/>intent_examples.embedding<br/>(iterative_scan)"]
        txt["GIN ts_rank on<br/>intent_examples.tsv"]
    end

    emb --> vec
    tsq --> txt
    vec --> rrf["RRF fusion<br/>1 / (60 + rank)"]
    txt --> rrf
    rrf --> agg["max score per intent<br/>top ROUTING_CANDIDATES"]
    agg --> sel["Router LLM<br/>select <= ROUTING_MAX_INTENTS<br/>or no_match"]
    sel --> out["selected intents<br/>+ entities"]
```

---

## 8. Capabilities and bindings

Playbook steps refer to abstract capabilities. Each company decides how, or whether, each one is implemented.

```mermaid
flowchart LR
    step["playbook step<br/>type=lookup<br/>capability=order.get"] --> reg["capability registry<br/>name, kind, schemas, policy keys"]
    reg --> bind{"company binding<br/>status = live?"}
    bind -- "no" --> manual["manual step for agent"]
    bind -- "yes" --> gw["gateway:<br/>validate args, policy (writes),<br/>inject tenant + secret,<br/>timeout, SSRF guard,<br/>truncate, audit"]
    gw --> mcpA["mcp adapter<br/>mapped tool only"]
    gw --> httpA["http adapter<br/>templated request"]
    gw --> sqlA["sql_readonly adapter<br/>admin SELECT, READ ONLY tx"]
    gw --> mockA["mock adapter<br/>demo / tests"]
```

```mermaid
stateDiagram-v2
    [*] --> Draft: PUT binding
    Draft --> Tested: POST test (dry run ok)
    Tested --> Draft: config updated
    Tested --> Live: POST go-live
    Live --> Draft: config updated
    Live --> [*]: DELETE
    note right of Live
        Only live bindings are used
        by chats and action execution.
    end note
```

Proposed action lifecycle:

```mermaid
stateDiagram-v2
    [*] --> Denied: policy check fails at proposal
    [*] --> Proposed: policy ok
    Proposed --> Rejected: agent rejects
    Proposed --> Executing: agent approves (row locked)
    Executing --> Executed: adapter ok
    Executing --> Failed: adapter error / policy re-check fails
    Denied --> [*]
    Rejected --> [*]
    Executed --> [*]
    Failed --> [*]
```

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

    subgraph ci["GitHub Actions"]
        j1["job 1: ruff, mypy, unit"]
        j2["job 2: integration"] --> pgci[("pgvector/pgvector:pg17<br/>service container")]
    end

    subgraph prod["Production"]
        lb["load balancer"] --> a1["API replica"]
        lb --> a2["API replica"]
        a1 --> hs[("Hosted Supabase<br/>session pooler / direct :5432")]
        a2 --> hs
    end
```

The API process is stateless (all state lives in Postgres), so it scales horizontally. Only env vars differ between environments.

---

## 11. Key decisions

| # | Decision | Why | Alternatives considered | Trade-off |
|---|---|---|---|---|
| 1 | **Compile guidelines into structured playbooks at upload** | Guidelines are runbooks for humans. Turning them into intents, typed steps, rules and facts makes routing exact, makes steps citable, and links steps to tools. | Plain RAG over chunks (v1) | Needs an LLM compile step plus validation; mistakes there need a coverage check and an optional human review |
| 2 | **pgvector in the same Postgres; Chroma removed** | One store, one transaction, one backup. Tenant and version filters in SQL; cascade deletes. Hundreds of intents per company is small for pgvector. | Chroma, a dedicated vector DB | At very large scale, pgvector needs partitioning; `SearchGateway` keeps a swap possible |
| 3 | **Hybrid search (vector + full-text, RRF)** | Embeddings miss product names, codes and exact phrasing; keyword search catches them | Vector only | Two queries per search; both indexed and cheap |
| 4 | **Embed synthetic example messages per intent; score by best match** | Customer wording rarely matches an intent's name or description | Embedding the intent description | More rows and embedding cost at ingest |
| 5 | **Two-stage routing: search candidates, then a cheap LLM selects** | Scales to hundreds of intents (the router sees about 20); handles messages with several intents and no-match | Single classifier over all intents; vector top-1 | One extra (fast) LLM call per chat |
| 6 | **Company rules always injected, never retrieved** | Tone, sign-off and prohibitions apply to every reply; retrieval could drop them | Retrieve as chunks | Slightly larger prompt |
| 7 | **Reference facts as a separate retrieval index** | Prices, device lists and URLs aren't procedures and don't map to steps | Folding facts into intents | A second search per chat (runs in parallel) |
| 8 | **Stable intent keys across versions** | Metrics, evals, feedback and history stay comparable after re-uploads | New keys per version | Compiler must be given previous keys; covered by a test |
| 9 | **Immutable versions; single active pointer on `companies`** | Exactly one active version, instant rollback with no re-indexing | `is_active` flag on versions | Activation needs a row lock (`FOR UPDATE`) |
| 10 | **Auto-approve by default, gated by a retrieval eval** | Zero friction for normal uploads, but a poor compile doesn't go live silently | Always manual review; always auto | Gate threshold needs tuning per domain |
| 11 | **Routing evals keyed by intent key, run per version** | Lets you compare v1 and v2 before a rollback; agent feedback becomes test data | Ad-hoc manual testing | Synthetic cases can be easier than real traffic, which feedback cases offset |
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

### Out of scope (future work)
Multi-turn threads, SSO/RBAC, a queue worker, multilingual messages, PDF/DOCX/OCR parsers (the interface exists), native OAuth connectors (Shopify, Stripe, Zendesk) and OpenAPI import, KMS/Vault, per-company tool rate limits, cross-encoder re-ranking, LLM-judge reply evals, partitioning by company, streaming responses.
