# Run every target from the repository root.
# Compose reads .env from the repo root (--project-directory .).

COMPOSE  := docker compose --project-directory . -f deploy/compose/docker-compose.yml
ALEMBIC  := uv run alembic -c infrastructure/database/alembic.ini
PROFILES ?= core

.DEFAULT_GOAL := help
.PHONY: help setup build config up down ps logs psql cypher \
        db-migrate db-downgrade db-current db-history db-revision test-integration

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## Install every workspace package + dev tools exactly as locked
	uv sync --all-packages --frozen

build: ## Build the service images (vision, diagnosis)
	COMPOSE_PROFILES=apps $(COMPOSE) build

config: ## Show the resolved compose file (passwords included: do not paste it anywhere)
	COMPOSE_PROFILES=$(PROFILES) $(COMPOSE) config

up: ## Start PROFILES (default core; PROFILES=apps adds the APIs) and wait until healthy
	COMPOSE_PROFILES=$(PROFILES) $(COMPOSE) up -d --wait

down: ## Stop and remove all containers (volumes are kept)
	COMPOSE_PROFILES='*' $(COMPOSE) down

ps: ## Container status
	COMPOSE_PROFILES='*' $(COMPOSE) ps

logs: ## Follow logs (make logs s=neo4j for one service)
	COMPOSE_PROFILES='*' $(COMPOSE) logs -f $(s)

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

test-integration: ## Integration tests (testcontainers: needs Docker, not the compose stack)
	uv run pytest -m integration
