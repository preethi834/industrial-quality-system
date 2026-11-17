# Implementation plan

Persistent checklist and session hand-off (blueprint §7.6). A new session starts here, then runs
`make setup && make check`.

Status labels (blueprint §7.1): **planned** · **in progress** · **working (tested)** · **known issue**.

## Roadmap

| Step | Scope (blueprint §5) | Status |
| --- | --- | --- |
| 0 | Foundation: repo, settings, logging, databases, service skeletons, CI, docs | **working (tested) locally**; known issue: CI not yet run on GitHub |
| 1 | P2 L1: defect detector (PatchCore / EfficientAD + supervised baseline) + demo | planned |
| 2 | P2 L2–3: MVTec AD 2 / VisA + conformal threshold | planned |
| 3 | P1 L1–2: diagnosis agent (hybrid RAG + KG, MCP, 8D) + evaluation | planned |
| 4 | P1 L3: conformal cause sets | planned |
| 5 | Factory simulator + scientist agent | planned |
| 6 | Cost-aware accept / reject / review | planned |
| 7 | Integrated quality loop (vision event → diagnosis case) | planned |
| 8 | Advanced features (8a–8g), one at a time | planned |
| 9 | Public proof of work (benchmark release, report, video) | planned |

## Step 0 — Foundation

| Sub-step | Content | Commit | Status |
| --- | --- | --- | --- |
| 0a | uv workspace (six packages), CPU-only torch index, Linux x86_64 lock, repo skeleton | `b6c666f` | working (tested) |
| 0b | `iqs_common`: settings, logging, db, ids, taxonomy model, events | `51c0ea0` | working (tested) |
| 0c | Cause taxonomy v0.1.0 (10 causes), config test, AI4I labelling guide | `516a244` | working (tested) |
| 0d | Compose (Postgres 16 + pgvector, Neo4j 5.26), Alembic baseline, outbox model | `98baecc` | working (tested) |
| 0e | Vision + diagnosis skeletons: `/health`, `/ready`, `/v1/capabilities`, API key, Docker images | `18c3700` | working (tested) |
| 0f | CI workflows, local pre-commit hooks, single mypy command | `3583846` | working locally; known issue: not run on GitHub |
| 0g | Docs: architecture, plan, decisions, experiments, testing, known issues, changelog | this commit | working |

Blueprint "done when" for step 0:

- [x] `make setup && make test` works; `make up` starts the stack on a clean checkout (+ `.env`)
- [x] both services answer `/health` (and `/ready`) in Docker — verified 2026-11-17
- [x] docs exist with real content
- [ ] green CI on `main` — needs a push (KNOWN_ISSUES #1, #2)

## Hand-off notes

### 2026-11-17 — end of step 0 (0d–0g)

- **Done:** 0d–0g committed locally on branch `step-0` (not pushed). 121 unit/API tests and 15
  integration tests pass; one `mypy` run over 38 files is clean; all 13 pre-commit hooks pass;
  compose `apps` profile passes `scripts/smoke_test.sh`.
- **Broken / open:** CI never ran on GitHub; actions not SHA-pinned; one unexplained integration
  failure (1 of 6 runs). See `KNOWN_ISSUES.md`.
- **Local state:** `.env` exists (git-ignored, random passwords and API key); Docker volumes
  `iqs_pg_data`, `iqs_neo4j_data`, `iqs_neo4j_logs` hold migration `0001`; images
  `iqs-vision:dev`, `iqs-diagnosis:dev` are built; stack is stopped.
- **Next:** push `step-0`, get CI green, merge to `main`, tag `v0.1-step0`; then step 1.
