# Run every target from the repository root.
# Compose reads .env from the repo root (--project-directory .).

COMPOSE  := docker compose --project-directory . -f deploy/compose/docker-compose.yml
ALEMBIC  := uv run alembic -c infrastructure/database/alembic.ini
PROFILES ?= core

.DEFAULT_GOAL := help
.PHONY: help setup build config up down ps logs logs-once psql cypher smoke \
        db-migrate db-downgrade db-current db-history db-revision \
        lint typecheck test test-integration test-e2e up-loop down-e2e pre-commit check

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## Install every workspace package + dev tools exactly as locked
	uv sync --all-packages --frozen

build: ## Build the images (vision, diagnosis, dashboard)
	COMPOSE_PROFILES=apps $(COMPOSE) build

config: ## Show the resolved compose file (passwords included: do not paste it anywhere)
	COMPOSE_PROFILES=$(PROFILES) $(COMPOSE) config

LOOP_COMPOSE := $(COMPOSE) -f deploy/compose/docker-compose.loop.yml

up: ## Start PROFILES (default core; PROFILES=apps adds the APIs) and wait until healthy
	COMPOSE_PROFILES=$(PROFILES) $(COMPOSE) up -d --wait

up-loop: ## Quality-loop stack: apps + vision/diagnosis workers (needs IQS_LOOP_DIR content)
	COMPOSE_PROFILES=apps $(LOOP_COMPOSE) up -d --wait

down: ## Stop and remove all containers (volumes are kept)
	COMPOSE_PROFILES='*' $(LOOP_COMPOSE) down --remove-orphans

ps: ## Container status
	COMPOSE_PROFILES='*' $(LOOP_COMPOSE) ps

logs: ## Follow logs (make logs s=neo4j for one service)
	COMPOSE_PROFILES='*' $(COMPOSE) logs -f $(s)

logs-once: ## Print recent logs of all services (used by CI on failure)
	COMPOSE_PROFILES='*' $(COMPOSE) logs --tail 200

smoke: ## Smoke test the running apps profile (/health, /ready, API key)
	scripts/smoke_test.sh

psql: ## psql shell in the Postgres container
	$(COMPOSE) exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

cypher: ## cypher-shell in the Neo4j container
	$(COMPOSE) exec neo4j sh -c 'cypher-shell -u "$${NEO4J_AUTH%%/*}" -p "$${NEO4J_AUTH#*/}"'

db-migrate: ## Apply all migrations (alembic upgrade head)
	$(ALEMBIC) upgrade head

db-downgrade: ## Revert one migration (make db-downgrade rev=base for all)
	$(ALEMBIC) downgrade $(or $(rev),-1)

db-current: ## Show the applied revision
	$(ALEMBIC) current

db-history: ## List migrations
	$(ALEMBIC) history

db-revision: ## Autogenerate a migration: make db-revision m="add cases table"
	@test -n "$(m)" || { echo 'usage: make db-revision m="message"'; exit 1; }
	$(ALEMBIC) revision --autogenerate -m "$(m)"

lint: ## ruff check + format check
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

typecheck: ## mypy over all trees (config in pyproject.toml)
	uv run --frozen mypy

test: ## Unit + API tests (no Docker needed)
	uv run --frozen pytest

pre-commit: ## All pre-commit hooks on all files (what CI runs)
	uv run --frozen pre-commit run --all-files

check: pre-commit test ## Everything CI's first job runs

test-integration: ## Integration tests (testcontainers: needs Docker, not the compose stack)
	uv run pytest -m integration

test-e2e: ## Step-7 quality loop on its own compose project iqs-e2e (make build first; make down first on the 10 GB VM)
	uv run --frozen pytest -m e2e tests/e2e -s

down-e2e: ## Remove a kept e2e stack (IQS_E2E_KEEP=1) and its volumes
	docker compose -p iqs-e2e down -v --remove-orphans
