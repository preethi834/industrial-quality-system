# Known issues

Limitations, open items and unexplained behaviour. Each entry: what, impact, next action.
Status at the end of step 0 (2026-11-17).

## Open

| # | Issue | Impact | Next action |
| --- | --- | --- | --- |
| 1 | **CI has never run on GitHub.** `ci.yml` and `docker.yml` are written and their commands were run locally, but nothing is pushed. | Step 0 "done when: green CI on main" is not met yet. | Push the branch, open a PR, fix anything runner-specific. |
| 2 | **Actions pinned by major tag** (`actions/checkout@v4`, `astral-sh/setup-uv@v6`), not by commit SHA; versions not re-checked against GitHub. | A moved tag could change CI behaviour. | Before the first push: look up current versions and pin to SHAs. |
| 3 | **One unexplained integration-test failure** (1 of 6 runs of `make test-integration`, 2026-11-17, right after `make down`); the output was not captured; 5 later runs, including the same sequence, passed. | Possible flakiness in container start-up. | If it recurs, keep the full log (`pytest -m integration -rA`); CI logs will capture it. |
| 4 | **Windows PostgreSQL 17 on port 5432** on the development machine. Compose publishes Postgres on **5433**. | Tools that assume 5432 from WSL reach the Windows server, not the container. | Keep `IQS_DATABASE__PORT=5433`. |
| 5 | **`uv sync` without `--all-packages` uninstalls the workspace packages** (root project is `package = false`). | Imports of `iqs_vision`, `torch`, ... break after a plain `uv sync`. | Always use `make setup` (`uv sync --all-packages --frozen`). |
| 6 | **Docker build cache ≈ 12.5 GB** after the first image builds (uv cache mount + layers). | Disk use in the WSL VM. | `docker builder prune` when disk is tight (next build re-downloads wheels). |
| 7 | **torch (CPU, 814 MB unpacked) is in both images**; diagnosis needs it via sentence-transformers. | Images 616 / 718 MB compressed. | Revisit in step 3 when the embedding model is chosen. |
| 8 | **Profile naming differs from the blueprint**: blueprint §3 names `core / vision / llm / observability`; step 0 uses `core` and `apps` (both APIs). | Cosmetic, until vision gets heavy extras. | Decide in step 1 whether vision moves to its own `vision` profile. |
| 10 | **detect-secrets 1.5.0** is the newest release but dates from May 2024. | Detector rules may lag newer token formats. | Re-evaluate (e.g. gitleaks in CI only) if it misses something. |
| 11 | **One sample of the diagnosis container at 47 % CPU** while idle, taken ~20 s after Postgres was restarted; all other samples were idle. | Probably reconnect work; not confirmed. | Re-measure with `docker stats` over a minute in step 3. |

## Resolved during step 0

- Leftover empty `apps/{diagnosis,factory-simulator,vision}/` directories → removed.
- Host port 5432 clash with the Windows PostgreSQL → moved to 5433 (ADR-001).
- mypy "Source file found twice under different module names" → `mypy_path` + `files` (ADR-003).
- FastAPI treated the API-key header as a query parameter (local variable in `Annotated` under
  postponed annotations) → module-level `APIKeyHeader` scheme; covered by tests.
