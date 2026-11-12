# Support Guidance Assistant — developer commands.
# Uses the Supabase CLI if installed, otherwise falls back to `npx supabase`.
SUPABASE ?= $(shell command -v supabase >/dev/null 2>&1 && echo supabase || echo npx supabase)
UV ?= uv
RUN = $(UV) run

.PHONY: help install db-up db-down db-reset db-status migrate seed run test-unit test-integration test test-live lint format typecheck check

help:
	@echo "install | db-up | db-down | db-reset | db-status | migrate | seed | run"
	@echo "test-unit | test-integration | test | test-live | lint | format | typecheck | check"

install:
	$(UV) sync

db-up:          ## start local Supabase + create the integration-test database
	$(SUPABASE) start
	$(RUN) python scripts/create_test_db.py

db-down:
	$(SUPABASE) stop

db-status:
	$(SUPABASE) status

db-reset:       ## wipe local DB, re-apply Alembic schema, re-seed
	$(SUPABASE) db reset
	$(RUN) alembic upgrade head
	$(RUN) python scripts/create_test_db.py
	$(RUN) python scripts/seed.py

migrate:
	$(RUN) alembic upgrade head

seed:
	$(RUN) python scripts/seed.py

run:
	$(RUN) uvicorn app.main:app --reload

test-unit:
	$(RUN) pytest -m unit

test-integration:
	$(RUN) pytest -m integration

test:
	$(RUN) pytest -m "unit or integration"

test-live:
	$(RUN) pytest -m live

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck:
	$(RUN) mypy app

check: lint typecheck test
