# Step 7 — connect the projects: the integrated quality loop (plan)

Status: **7a–7h done locally, not pushed.** Gates 1, 2, 4–7 met; **gate 3 met for the plumbing only**: the real-image demo runs end to end, but its alert came from false alarms on good images (KNOWN_ISSUES #46; after fix (a)+(c) still 5 of 6 alerting images are goods). (7g/7h 2026-11-22; 7a–7f 2026-11-21,
branch `step-7` from `main` at `c89c9af`). Plan approved by the
owner on 2026-11-21 with decisions D1–D8 as recommended (§10); D2 (read-only symlinks) on 2026-11-22.
Implementation log and deviations from this plan: §12.

Sources read: blueprint Rev 2 §1.4, §2.1–2.3, §2.6–2.9, §5 (step 7, step 8 table, sequencing
note); `docs/IMPLEMENTATION_PLAN.md`; `docs/ARCHITECTURE.md`; `docs/KNOWN_ISSUES.md`;
`docs/notes/step-{3,5a,6}.md`; the code in `iqs_common.{events,outbox,ids,api}`,
`iqs_diagnosis.{agent,api,persistence,ingestion,domain}`, `iqs_vision.{api,detectors,testing}`,
`iqs_sim.{export,cases}`.

---

## 1. Objective

When the vision system sees a **new, unknown kind of defect** (a cluster of anomalous images
that matches no known defect class), the diagnosis agent **automatically opens a root-cause
case**, runs to a ranked cause list + conformal cause set + 8D draft + QMS ticket, and waits
for the engineer. When the engineer **confirms** the cause, the confirmation flows back:

- diagnosis side: Neo4j `(:Case)-[:CONFIRMED_CAUSE]->(:Cause)` and
  `(:DefectType)-[:INDICATES]->(:Cause)`, a Dirichlet count per (defect type, cause), and the
  case goes into a **candidate pool** for the next versioned calibration set (never into the
  active one, blueprint §2.8 A5);
- vision side: the cluster becomes a **named defect class**, so later images of the same kind
  are recognised as known and do **not** open a second case.

Blueprint deliverable: "end-to-end quality loop (Tier A complete)", on a **core-only** install
(no optional extras, no LLM, no downloads).

### Where step 7 starts from (honest dependency state)

| Blueprint dependency | State on `main` today | Consequence for step 7 |
| --- | --- | --- |
| Step 1 detector + inspection API | working | reused read-only |
| Step 2 conformal threshold | **in progress on `step-2`** | the trigger uses whatever threshold the registry reports (today: provisional max-of-calibration). No false-alarm guarantee is claimed for the trigger. Gate re-run after step 2 merges. |
| Step 3 agent | minimum version (simulator symptom profiles, no LLM, **no Neo4j graph yet — 3g planned**) | step 7 adds only the minimal graph writes the gate needs (§5.6); full 3g graph stays planned |
| Step 4 conformal cause sets | working on simulator alarm cases | reused; vision-triggered cases are **not** exchangeable with the calibration cases (§9, KNOWN_ISSUES #41) |
| Step 5 simulator + scientist | working | reused; simulator gains `linker.py` |
| Step 6 cost decisions | **planned, not built** | not used; the trigger only needs "anomalous / not". KNOWN_ISSUES #42 |
| BN (step 8a) | not built | "BN count" in the gate = a Dirichlet count table that 8a will consume; step 7 does **not** feed it into scoring |

---

## 2. Event flow

```text
 SIMULATOR (world, synthetic)            VISION SERVICE                         DIAGNOSIS SERVICE
 ───────────────────────────             ──────────────                         ─────────────────
 scenario YAML ─► run_scenario
   observed/  ground_truth/
        │
 linker.py: part_id ─► image
 (synthetic linkage manifest)
        │  images in part order
        ▼
 demo / e2e driver ── POST /v1/discovery/inspect (image, category, part_id, run_id) ──►
                                          ① PatchCore score vs registry threshold
                                          ② anomalous? ─no─► done (observation stored)
                                          ③ embedding of the top-anomaly patches
                                          ④ near a NAMED class centroid? ─yes─► "known:<name>"
                                          ⑤ else ─► unknown buffer (vision.discovery_observations)
                                          ⑥ HDBSCAN over the category's buffer
                                          ⑦ new cluster ≥ min size, not yet alerted:
                                             ONE transaction: vision.clusters row
                                             + platform.events row  new_defect_cluster
                                                                │
                                                                │ outbox (Postgres), at-least-once
                                                                ▼
                                                       diagnosis worker (poll, SKIP LOCKED)
                                                       ⑧ ONE transaction: processed_events(event_id)
                                                          + diagnosis.cases row (id from cluster_id)
                                                       ⑨ agent.run(case) → awaiting_confirmation
                                                          (symptoms from observed/, retrieval,
                                                           novelty, scores, scientist, conformal set,
                                                           8D, MCP create_ticket)
                                                                │
 engineer (dashboard / API) ── GET /v1/cases?source=vision, /cause-set, /hypotheses, /report-8d
                            ── POST /v1/cases/{id}/confirm {cause_code, engineer, defect_class_name?}
                                                                ▼
                                                       ⑩ confirm → close_ticket (MCP)
                                                       ⑪ feedback_update node:
                                                          Postgres tx: cause_counts += 1,
                                                          calibration_candidates row, feedback_applied,
                                                          + platform.events  case_confirmed
                                                          Neo4j (idempotent MERGE): Case, Cause,
                                                          DefectType, CONFIRMED_CAUSE, INDICATES
                                                                │ outbox
                                          ◄─────────────────────┘
                                          vision worker
                                          ⑫ ONE tx: processed_events + cluster → status
                                             "named", name, cause_code, centroid frozen
                                          ⑬ later images near the centroid → known class (step ④),
                                             no new cluster, no new case
```

Every arrow that crosses a service boundary goes through the outbox; no service calls the
other's HTTP API. The dashboard talks only to the APIs.

---

## 3. Interfaces and data contracts

### 3.1 Events (`iqs_common/events.py`, already exists — small extension)

`NewDefectClusterPayload` and `CaseConfirmedPayload` exist with `schema_version: 1` and
`extra="forbid"`. No producer or consumer has ever run, so I extend v1 instead of bumping
(stated in ADR-015). Additions, all optional so existing tests keep passing:

| Field | On | Meaning |
| --- | --- | --- |
| `line_id: str \| None` | NewDefectCluster | simulator line (synthetic linkage) |
| `simulator_run_id: str \| None` (pattern `run-[0-9a-f]{12}`) | NewDefectCluster | which run's `observed/` holds the linked parts; **synthetic linkage**, never set from real data |
| `linkage: Literal["synthetic_simulator", "none"] = "none"` | NewDefectCluster | provenance flag, shown in the case description and the 8D D2 text |
| `representative_inspection_id: UUID` | NewDefectCluster | the member closest to the centroid (for the dashboard thumbnail) |
| `score_max`, `score_median: float` | NewDefectCluster | anomaly scores of the members |
| `diagnosis_case_id: str` | CaseConfirmed | already `case_id: UUID` — **mismatch**: diagnosis case IDs are strings (`String(64)`, simulator IDs are not UUIDs). For vision cases I make the case ID a UUIDv5 of the cluster, so it *is* a UUID and fits the existing field; no change needed. Recorded so nobody "fixes" it later. |

Contract tests (§7.1) pin the JSON schema of both events in `tests/contracts/` and are run by
**both** services' test suites.

### 3.2 Outbox worker (`iqs_common/outbox_worker.py`, new)

```python
def claim(session, types: set[EventType], limit: int) -> list[OutboxEvent]
    # SELECT ... WHERE status='pending' AND type IN (...) ORDER BY created_at
    # FOR UPDATE SKIP LOCKED LIMIT :limit
def run_forever(factory, handlers: dict[EventType, Handler], poll_s=1.0, max_attempts=5)
    # per row: handler(session, event) in the SAME transaction as status='processed';
    # on exception: rollback, attempts += 1 (own tx), status='failed' at max_attempts, log
```

One consumer per event type, so the single `status` column on `platform.events` is enough;
if a second consumer of the same type ever appears, this needs a per-consumer delivery table
(noted in ADR-015, not built).

### 3.3 Idempotency rules

| Where | Key | Mechanism |
| --- | --- | --- |
| vision → alert | cluster | a cluster is alerted once: `vision.clusters.alerted_event_id` set in the same tx as the outbox row; a re-clustered group that overlaps an alerted cluster by ≥ 50 % of members is treated as the same cluster (it grows; no new event) |
| diagnosis consumer | `event_id` | `diagnosis.processed_events (event_id PK)`, inserted in the same tx as the case; duplicate → no-op |
| diagnosis consumer | `cluster_id` | case ID = `deterministic_id("vision-cluster", cluster_id)`; `cases.source_cluster_id` UNIQUE — a *different* event for the same cluster still cannot create a second case |
| agent run after consumer | case phase | worker runs the agent only if `phase == not_started`; a startup sweep re-runs vision cases left `registered` by a crash between ⑧ and ⑨ |
| feedback | `case_id` | `diagnosis.feedback_applied (case_id PK)` in the Postgres tx; Neo4j writes are `MERGE` (re-runnable) and retried until a `graph_written_at` timestamp is set |
| vision consumer | `event_id` | `vision.processed_events (event_id PK)` + cluster status transition `alerted → named` only |

### 3.4 HTTP (all under `/v1`, API key)

Vision (new router `iqs_vision/api/routes/discovery.py`):

| Route | Body / params | Returns |
| --- | --- | --- |
| `POST /v1/discovery/inspect` | multipart: `image`, `category`, `part_id?`, `run_id?`, `line_id?` | `{inspection_id, score, threshold, decision, routing: "normal" \| "known:<name>" \| "unknown_buffered", cluster_id?, alert_event_id?}` |
| `GET /v1/clusters` | `category?`, `status?` | clusters with size, status, name, cause, first/last seen |
| `GET /v1/clusters/{id}` | — | cluster + member inspection IDs |

Why a separate inspect route instead of hooking `POST /v1/inspections`: `inspections.py` is
being changed on the `step-2` branch. The discovery route reuses the same registry, decode
and detector calls (no copy of business logic beyond ~20 lines of request handling). After
step 2 merges, `POST /v1/inspections` can call the same `route_observation()` function in one
line — listed as a merge follow-up, not done here.

Diagnosis (existing `api/cases.py`, extended):

| Route | Change |
| --- | --- |
| `GET /v1/cases` | **new**: list with `source`, `status`, `cluster_id` filters (e2e "exactly one case") |
| `GET /v1/cases/{id}` | add `source`, `status`, `source_cluster_id`, `linkage` |
| `POST /v1/cases/{id}/confirm` | body gains optional `defect_class_name` (vision cases only; 422 otherwise); response gains `feedback` (counts before/after, graph written, event id) |
| `GET /v1/feedback/counts` | **new**: Dirichlet counts table (read-only, for the gate and the dashboard) |

### 3.5 Capabilities (gate 2)

`/v1/capabilities` today returns `features: dict[str, bool]`. Gate 2 needs the four states of
blueprint §2.9. Change (backward compatible): add `status: dict[str, "enabled" | "disabled" |
"not_installed" | "missing_artifact"]` and `reason: dict[str, str]` next to the existing
booleans. New flags: vision `discovery_trigger` (step 7; `discovery` stays the step-8c flag),
diagnosis `vision_cases`, `feedback`, `graph` (true only when Neo4j writes are configured).

---

## 4. Data model (one migration `0005_quality_loop`)

| Table | Columns (main) |
| --- | --- |
| `vision.discovery_observations` | `id` (inspection UUIDv7, PK), `category`, `part_id?`, `run_id?`, `score`, `threshold`, `decision`, `routing`, `embedding real[]` (d ≈ 1536, see D1), `cluster_id?`, `model_version`, `created_at` |
| `vision.clusters` | `id` UUIDv7 PK, `category`, `status` (`alerted` / `named`), `member_count`, `centroid real[]`, `radius`, `first_seen`, `last_seen`, `alerted_event_id`, `defect_class_name?`, `cause_code?`, `case_id?`, `named_at?` |
| `vision.processed_events` | `event_id` PK, `processed_at` |
| `diagnosis.cases` | + `source_cluster_id UUID NULL UNIQUE` |
| `diagnosis.processed_events` | `event_id` PK, `case_id`, `processed_at` |
| `diagnosis.cause_counts` | `context` (e.g. `defect_type:<category>/<name>`), `cause_code`, `count` (int), PK (context, cause_code). Prior α₀ = 1 per cell is implicit (not stored). |
| `diagnosis.calibration_candidates` | `case_id` PK, `source`, `cause_code`, `added_at`, `used_in_calibration_run?` (stays NULL in step 7) |
| `diagnosis.feedback_applied` | `case_id` PK, `applied_at`, `graph_written_at?`, `event_id?` |

Embeddings are `real[]`, not pgvector: nearest-centroid and HDBSCAN run in Python on ≤ a few
hundred rows per category; pgvector indexing earns nothing here.

**Migration numbering risk:** The `step-2` branch may add a `0005`. Mine is `0005` with
`down_revision = 0004`; whichever merges second renumbers (KNOWN_ISSUES #45).

---

## 5. Modules: reused vs new

### 5.1 Reused unchanged

`iqs_common.{ids,db,outbox,taxonomy,log}`; `iqs_vision.{inference.registry,data.preprocess,
detectors.patchcore,testing}` (read-only use); `iqs_diagnosis.{agent nodes intake→close_ticket,
ingestion.simulator.load_case pattern, symptoms, retrieval, scoring, novelty, conformal,
scientist, reporting.eight_d, persistence}`; `iqs_mcp` (in-process, as today); `iqs_sim.
{pipeline,scenarios,export}`.

### 5.2 Simulator — `iqs_sim/linker.py` (new) + `configs/simulator/image_linkage.yaml` (new)

- Input: a run directory, an image pool (directory of good images + per-defect-type folders,
  MVTec layout), and the linkage config `fault_type → defect_type` per product category.
- For each part in time order: if `ground_truth/part_truth.true_surface_defect` is true and an
  active fault maps to a defect type → draw an image of that type; otherwise a good image.
  Deterministic (named RNG stream `linker`), sampling with replacement, recorded in
  `<run>/linkage/manifest.parquet` (part_id, timestamp, image path, `linkage="synthetic"`).
- The linker is part of the **world**, like the fault injector, so it may read
  `ground_truth/`. The manifest's defect-type column is ground truth too and lives under
  `<run>/linkage/` which no service mounts (access test, §7.1).
- Only faults with a surface effect are mappable (tool wear, overload, cooling, drive,
  parameters, bad batch, ambient); dimension-only faults (fixture, handling, gauge drift)
  produce no visual defect, which the config enforces (validation error).
- Linkage for the e2e test: `iqs_vision.testing` synthetic images (stripe goods; `scratch`
  defects). Linkage for the demo: MVTec images (only after approval of read-only symlinks, D2).

### 5.3 Vision — discovery trigger (new, core-only)

`iqs_vision/discovery/` — the blueprint names `trigger.py`; I split it so the domain part is
pure and unit-testable:

| File | Role |
| --- | --- |
| `embed.py` | region embedding: mean of the PatchCore backbone patch features over the top-k anomaly-map patches, L2-normalised (D1) |
| `trigger.py` | pure domain: `assign_known(emb, named_clusters)`, `find_new_clusters(buffer, alerted, params)` (sklearn `HDBSCAN`, `min_cluster_size`, overlap rule), cluster centroid + radius (95th percentile member distance) |
| `store.py` | SQLAlchemy rows + repository for observations / clusters / processed events; writes cluster + outbox event in one tx |
| `service.py` | application: `route_observation(image, meta)` = score → threshold → embed → known/unknown → maybe alert |
| `consumer.py` | handler for `case_confirmed` (cluster → named) |
| `worker.py` | `python -m iqs_vision.discovery.worker` (outbox loop for vision's event types) |
| `api/routes/discovery.py` | the three routes in §3.4 |

Trigger parameters (`configs/vision/discovery.yaml`): `min_cluster_size = 5`, `min_samples = 3`,
overlap 0.5, known-class radius factor 1.0, `top_k_patches = 10`. These are **stated starting
values, not tuned or evaluated** (blueprint: "no evaluation claims" for the minimal trigger;
8c evaluates).

### 5.4 Diagnosis — consumer, vision cases, feedback (new)

| File | Role |
| --- | --- |
| `ingestion/vision.py` | `case_from_cluster_event(event, runs_root) -> DiagnosisCase`: `source="vision"`, deterministic case ID, title/description from the payload (with the linkage flag), `observed_dir = runs_root/<simulator_run_id>/observed`, `window_start = first_seen`, `alarm_ts = last_seen`, `baseline_hours` from config (12 h, same as the simulator alarm rule), `reported_symptoms = ({"code": "new_visual_defect_cluster", ...},)` |
| `worker/consumer.py` | handler for `new_defect_cluster` (tx: processed_events + case) then `agent.run`; startup sweep |
| `worker/__main__.py` | `python -m iqs_diagnosis.worker` |
| `feedback/update.py` | `apply_feedback(case, confirmation)`: Postgres tx (counts, candidate pool, feedback_applied, `case_confirmed` outbox row for vision cases) + `GraphWriter` calls |
| `graph/repository.py` | minimal `GraphWriter` port + Neo4j adapter: `MERGE (c:Case {id})`, `(k:Cause {code})`, `(d:DefectType {category, name})`, `CONFIRMED_CAUSE`, `INDICATES {count}`; in-memory fake for unit tests. This is a **subset of 3g**, not the 3g graph |
| `agent/graph.py` | **one new node** `feedback_update` after `close_ticket` (runs alone after the interrupt, as the module's rule requires); `AgentDeps` gains `feedback: FeedbackService \| None` (None = node is a no-op, so existing tests are unchanged) |
| `symptoms` / `configs/diagnosis/symptoms.yaml` | add the `new_visual_defect_cluster` code (reported symptom; carries no feature, so the ranker is unchanged) |

Confirmation with `cause_code = null` ("no listed cause fits"): case is closed as out of
taxonomy, no count, no `CONFIRMED_CAUSE` edge, **no** `case_confirmed` event (its payload
requires a cause); the cluster stays `alerted` and is listed as "unresolved" in the dashboard.

### 5.5 Dashboard

New page `pages/5_Quality_Loop.py` (number 5 to stay clear of the `step-2` branch's pages 2 and 3):
open vision clusters → linked case → cause set + hypotheses + 8D link → confirm form (cause,
engineer, defect class name). Plus `iqs_dashboard/capabilities.py`: one helper that renders
"not installed — `uv sync --extra <x>`" for any feature whose status ≠ `enabled`; used by the
new page and the home page. Pages 2/3 are not touched.

### 5.6 Deployment

`deploy/compose/docker-compose.yml`: two services `vision-worker`, `diagnosis-worker` (same
images, different `command`) in the `apps` profile; a `runs` volume for simulator runs mounted
read-only into diagnosis (and the linkage dir mounted into **nothing**). The blueprint's gate
says "`--profile core`"; in this repo the blueprint's *core stack* is our `apps` profile (our
`core` = databases only, ADR-001). I'll write that mapping into the gate text instead of
renaming profiles.

### 5.7 Scripts

`scripts/demo_loop.py`: simulate → link → (fit or load model) → stream images → wait for the
case → print cause set → confirm (cause from `--confirm-cause` or, with `--oracle`, from
ground truth, labelled "oracle engineer") → print feedback → stream more images → show "known
class, no new case". Writes `reports/quality_loop/<run_id>.json` with the recorded metrics
(§8). Driven through the HTTP APIs only.

---

## 6. New and changed files

New:

```text
packages/simulator/src/iqs_sim/linker.py
configs/simulator/image_linkage.yaml
packages/common/src/iqs_common/outbox_worker.py
packages/vision/src/iqs_vision/discovery/{__init__,embed,trigger,store,service,consumer,worker}.py
packages/vision/src/iqs_vision/api/routes/discovery.py
configs/vision/discovery.yaml
packages/diagnosis/src/iqs_diagnosis/ingestion/vision.py
packages/diagnosis/src/iqs_diagnosis/worker/{__init__,__main__,consumer}.py
packages/diagnosis/src/iqs_diagnosis/feedback/{__init__,update}.py
packages/diagnosis/src/iqs_diagnosis/graph/{__init__,repository}.py
packages/dashboard/src/iqs_dashboard/capabilities.py
packages/dashboard/src/iqs_dashboard/pages/5_Quality_Loop.py
infrastructure/database/migrations/versions/0005_quality_loop.py
tests/contracts/{new_defect_cluster,case_confirmed}.schema.json
tests/e2e/{conftest,test_quality_loop}.py
scripts/demo_loop.py
(+ unit test files next to each package's existing tests)
```

Changed (small, listed so the overlap with the `step-2` branch is visible):

| File | Change | Overlap with `step-2`? |
| --- | --- | --- |
| `iqs_common/events.py` | optional payload fields (§3.1) | no |
| `iqs_common/api.py` | capability `status` / `reason` maps | no (A may read it) |
| `iqs_vision/api/main.py` | mount discovery router, `discovery_trigger` flag | **possible**: A may mount `calibrations.py` here. One `include_router` line each → trivial merge. Flagged. |
| `iqs_vision/detectors/patchcore.py` | public `patch_features(images)` (wraps `_embed`) | not on A's list; A may read it |
| `iqs_vision/settings.py` | `discovery` sub-settings | not on A's list; possible small conflict |
| `iqs_diagnosis/agent/graph.py`, `api/cases.py`, `api/wiring.py`, `api/main.py`, `settings.py`, `persistence/repository.py` | §5.4 | no |
| `configs/diagnosis/symptoms.yaml` | one symptom code | no |
| `deploy/compose/docker-compose.yml`, `Makefile` (`test-e2e`, `demo-loop`, `worker-*`) | §5.6 | no |
| `pyproject.toml` | pytest marker `e2e` | no |
| docs: ARCHITECTURE, TESTING, EXPERIMENTS, CHANGELOG, KNOWN_ISSUES (#40+), TECHNICAL_DECISIONS (ADR-015/016), IMPLEMENTATION_PLAN | at the end of the step | IMPLEMENTATION_PLAN / KNOWN_ISSUES / CHANGELOG are edited by A too → append-only sections, merge by hand |

Not touched: `iqs_vision/calibration/`, `inference/registry.py`, `api/routes/inspections.py`,
`api/routes/calibrations.py`, dashboard pages 2 and 3, `configs/assets.yaml`,
`docs/APPROVALS.md`. **No new dependency** (scikit-learn and timm are already vision deps);
if `uv.lock` changes at any point I stop and ask.

---

## 7. Tests

### 7.1 Unit (no Docker; run with `nice -n 19`, changed files only while A trains)

- **Events / contracts:** both payloads round-trip; JSON schema equals the pinned file in
  `tests/contracts/` (fails on any unannounced contract change); old v1 payloads without the
  new fields still parse.
- **Linker:** deterministic (same seed → same manifest hash); defective parts get images of
  the mapped type, good parts good images; dimension-only fault mapping is rejected; manifest
  rows = parts.
- **Trigger domain:** synthetic embeddings (two Gaussian blobs + noise): no alert below
  `min_cluster_size`; exactly one alert at the first point where a blob reaches it; a growing
  cluster does not re-alert (overlap rule); a point near a named centroid routes to
  `known:<name>`; noise points stay buffered.
- **Embedding:** shape and L2 norm; top-k patch selection follows the anomaly map (tiny
  random-init backbone, no weights).
- **Discovery service:** normal image → no buffer row; anomalous → buffer; 5th anomalous of
  one kind → one cluster row **and** one outbox row in the same session (in-memory/SQLite-free
  fake repository).
- **Vision case builder:** `case_from_cluster_event` → deterministic ID, `source="vision"`,
  correct `observed_dir`, linkage text present; missing run → clear error (event marked failed,
  not silently dropped).
- **Consumer idempotency (in-memory):** same event twice → one case; two different events for
  one cluster → one case; crash after case insert → sweep resumes the run once.
- **Feedback:** counts +1 exactly once (re-apply is a no-op); candidate pool row written;
  **active calibration file and its hash unchanged**; `case_confirmed` emitted only for vision
  cases with a cause; `cause_code=None` path; graph writer called with the expected MERGEs
  (fake).
- **Agent:** existing graph tests unchanged with `feedback=None`; new test with a fake
  feedback service: node runs once after confirm, not on resume replays.
- **API:** `GET /v1/cases` filters; `defect_class_name` rejected for non-vision cases (422);
  discovery routes 401 / 415 / 501 (feature disabled) / 200.
- **Capabilities + dashboard (gate 2):** status map values; the dashboard helper renders "not
  installed" for every non-enabled feature (Streamlit `AppTest`, as `test_dashboard_inspection.py`).
- **Access rule:** no diagnosis or vision module opens `ground_truth/` or `linkage/`
  (static grep test like the ADR-008 access test).

### 7.2 Integration (testcontainers; **only after the `step-2` training chain has finished**)

- Migration `0005` upgrade / downgrade / upgrade.
- Outbox `claim` with two concurrent sessions: `SKIP LOCKED` gives disjoint rows; failure →
  `attempts` increments, `failed` at the maximum.
- Diagnosis consumer on Postgres: duplicate delivery → one `cases` row, one
  `processed_events` row; `source_cluster_id` UNIQUE enforced.
- Neo4j `GraphWriter`: MERGE twice → one edge; `INDICATES.count` equals the Postgres count.
- Vision consumer: cluster `alerted → named`; second delivery no-op.

### 7.3 End-to-end (`tests/e2e/test_quality_loop.py`, marker `e2e`, against the compose `apps` stack; **only after A's chain**)

Scripted exactly as blueprint gate 1, on **simulator data with synthetic images**:

1. `run_scenario(tool_wear_m1)` (existing reference scenario; shortened duration if the full
   72 h is too slow — decided when measured) into the shared `runs` volume.
2. Fit a test PatchCore on `iqs_vision.testing` good images with a **random-init backbone**
   (no weight download), register it under a test category.
3. Link parts → synthetic images (tool wear → `scratch`).
4. Stream images in part order to `POST /v1/discovery/inspect`.
5. Poll `GET /v1/cases?source=vision` → **exactly one** case; record latency.
6. `GET /cause-set` returns a set (or the explicit empty/novel answer).
7. Re-insert the same event as `pending` → still exactly one case (idempotency in the stack).
8. `POST /confirm` with the true cause (the test harness may read ground truth; services may not).
9. Assert Neo4j `CONFIRMED_CAUSE` edge exists, `cause_counts` for that context went n → n+1,
   candidate-pool row exists, calibration run file hash unchanged, vision cluster `named`.
10. Stream more `scratch` images → routed `known:<name>`, still one case.

It proves the plumbing, **not** detection quality: random-init features on striped synthetic
images say nothing about MVTec performance. The test name and docstring say so.

### 7.4 Demo on real images (manual, gate 3)

`scripts/demo_loop.py` with a step-1 **deploy** PatchCore model and real MVTec AD defect
images, linked synthetically to simulator parts. Needs read-only symlinks to
`~/code/industrial-quality-ai/{data,models}` (D2) and A's chain to be finished. Output marked
"real images, synthetic process linkage, simulated process data".

---

## 8. Gates and recorded metrics

Gate (binary):

1. `tests/e2e/test_quality_loop.py` passes against the compose stack (`make up PROFILES=apps`
   + workers) built **without extras** (`EXTRAS` default), scripted as §7.3 steps 1–10.
2. Dashboard: every feature whose status ≠ `enabled` renders "not installed / disabled" with
   the reason (unit `AppTest`); no button calls an endpoint that returns 501.
3. `scripts/demo_loop.py` runs end to end on this machine (real images, §7.4) — or, if the
   owner declines the symlinks, the demo runs on synthetic images and gate 3 is recorded as
   "met on synthetic images only".
4. (added) Idempotency proven at three levels: unit, integration (Postgres), e2e (step 7).
5. (added) Confirmation leaves the active calibration run byte-identical (hash check) and
   writes the case only to the candidate pool.
6. (added) Contract tests run in both services' test suites.
7. Universal gate: `make check` green; docs updated; tag — CI on GitHub and
   tag only after the owner pushes (never by me).

Recorded (measured, with run ID, labelled by data type):

- wall-clock latency from the first anomalous image processed to the case being opened
  (and to `awaiting_confirmation`), median over ≥ 5 e2e/demo repeats;
- number of images inspected after fault onset, and number of anomalous images, before the
  alert (onset from ground truth, read only by the script);
- per run: cluster size at alert, false routing to `known` before confirmation (should be 0
  by construction), cases opened (should be 1).

No accuracy, purity or delay-distribution claims: one scenario × few repeats is a
demonstration, not an evaluation (8c evaluates discovery over ≥ 3 defect types × ≥ 3 seeds).

---

## 9. Honestly out of scope (and what this step does *not* show)

- **Real process linkage.** MVTec images carry no process data; part ↔ image linkage is
  synthetic and every vision case says so (KNOWN_ISSUES #40).
- **Guarantee for vision-triggered cases.** The conformal cause set is calibrated on
  simulator *alarm* cases; a cluster-triggered case is a different trigger and time window, so
  exchangeability is not established. The cause-set response for `source="vision"` will
  state "not covered by the calibration run" (KNOWN_ISSUES #41). A vision-source calibration
  run is future work.
- **Step 2 conformal threshold / step 6 cost decisions** — the trigger uses the registry
  threshold as it is; re-run the gate after those merge (KNOWN_ISSUES #42).
- **Evaluated discovery** (UMAP, stability, purity/ARI, labelling queue UI) — step 8c. Trigger
  parameters are untuned starting values (KNOWN_ISSUES #43).
- **Few-shot defect classes** — step 8d. "Vision learns" here means a named centroid with a
  radius rule, nothing more.
- **Bayesian network, EIG question loop** — step 8a. Counts are recorded but **not used** by
  the scorer in step 7 (KNOWN_ISSUES #44). The retrieval library is **not** changed by a
  confirmation; the case only enters the candidate pool (D3).
- **FMEA proposals** — step 8b. **Full Neo4j graph** (parts, processes, failure modes, FMEA)
  — step 3g; step 7 writes only Case / Cause / DefectType nodes.
- **LLM** (symptom extraction from text, German 8D) — 3e; not approved, not used.
- **Separate MCP service** — stays in-process as today.
- **Exactly-once delivery** — at-least-once + idempotent handlers, by design.
- **Multi-consumer outbox, dead-letter UI, auth between services** — not needed at this scale.

---

## 10. Decisions I need from you

| # | Question | My recommendation |
| --- | --- | --- |
| D1 | Embedding for clustering: (a) PatchCore backbone patch features (already loaded, no download) or (b) DINOv2 ViT-S/14 as in the blueprint (~85 MB weight download, approval gate, adds a second model in the vision container) | **(a)**, recorded as a deviation in ADR-015; DINOv2 can be compared in 8c |
| D2 | Real images for the demo (gate 3): later, read-only symlinks `data -> ~/code/industrial-quality-ai/data` and `models -> …/models` in this worktree | yes, but only after A's chain finishes; until then everything runs on synthetic images |
| D3 | Does a confirmed case enter the **live** retrieval library? | **no** in step 7: candidate pool only; the active library and calibration stay frozen (blueprint §2.8 A1/A5). A versioned rebuild is a later, explicit step |
| D4 | Minimal Neo4j writer in step 7 (subset of 3g) | yes — the gate requires the edge |
| D5 | Extend event schema v1 with optional fields instead of bumping to v2 | yes (no producer/consumer ever ran) |
| D6 | New dashboard page number `5_Quality_Loop.py` | yes |
| D7 | Touch `iqs_vision/api/main.py` and `detectors/patchcore.py` (small, §6) | yes, flagged for the merge |
| D8 | e2e scenario: `tool_wear_m1`, possibly shortened to keep the e2e under ~2 min | yes; the duration is chosen once measured and written down |

Proposed numbering: ADR-015 (quality-loop design: outbox worker, idempotency keys, embedding
choice, event v1 extension, profile mapping), ADR-016 (feedback policy: counts, candidate
pool, frozen calibration, graph subset). KNOWN_ISSUES #40–#45 as referenced above.

---

## 11. Sub-steps and order (each a local commit on `step-7` after approval)

| Sub-step | Content | Heavy? |
| --- | --- | --- |
| 7a | event fields + contract tests; `outbox_worker` (unit only) | light |
| 7b | `iqs_sim/linker.py` + config + tests | light |
| 7c | vision discovery domain (`embed`, `trigger`) + service + store models + route + vision consumer (unit) | light (tiny random-init backbone) |
| 7d | diagnosis vision-case builder, consumer, worker, `GET /v1/cases` (unit) | light |
| 7e | feedback service, graph writer (fake + Neo4j adapter), agent node, confirm extension (unit) | light |
| 7f | migration 0005, capabilities status, dashboard page + helper (unit) | light |
| 7g | integration tests, compose workers, e2e test, `make test-e2e` | **heavy — waits for your go after A's chain** |
| 7h | demo on real images (needs D2), metrics, docs, ADRs, KNOWN_ISSUES | **heavy — waits** |

Until you say A's chain is done: only unit tests, `nice -n 19`, changed files only; no Docker,
no testcontainers, no training, no image builds.

---

## 12. Implementation log (7a–7f) and deviations from the plan

| Sub-step | Commit | What |
| --- | --- | --- |
| 7a | `7a63f1a` | event v1 optional fields, pinned contracts in `tests/contracts/` checked by common, vision and diagnosis suites, `iqs_common.outbox_worker` (in-memory + SQL `SKIP LOCKED`) |
| 7b | `aa410a9` | `iqs_sim.linker` + `configs/simulator/image_linkage.yaml` |
| 7c | `2c871ad` | vision discovery: embedding, trigger, store, service, consumer, worker, routes |
| 7d | `552bf91` | diagnosis: vision case builder, consumer + worker, `GET /v1/cases`, guarantee flag |
| 7e | `b87cdfc` | feedback service + store, graph writer, `feedback_update` node, in-process loop test |
| 7f | `f61aca0` | migration 0005, capability status, dashboard page 5, these notes |
| 7g | `f941306` | compose workers (`docker-compose.loop.yml`), e2e stack (`docker-compose.e2e.yml`, project `iqs-e2e`), `tests/integration/test_quality_loop_db.py`, `tests/e2e/`, `make test-e2e` / `up-loop` / `down-e2e`, diagnosis Dockerfile fix, trigger fix found by the e2e |
| — | `b736c97` | merge of `origin/main` (step 2, PR #4): one conflict in `iqs_vision/api/main.py`, both routers kept |
| 7h | `331d000` | `scripts/demo_loop.py` (real-image demo + false-cluster measurement), KNOWN_ISSUES #40–#47, ADR-015/016 in the shared docs |
| 7h-fix | this commit | #46 owner decision (a)+(c): trigger counts distinct images (pixel SHA-256, migration 0006); metal_nut linkage = 66 unseen goods (calib + test), dealt without replacement; re-measured |

Deviations, each for a reason found while building:

1. **Trigger needed two extra rules** (`iqs_vision/discovery/trigger.py`). (a) A compactness
   gate `max_radius = 0.5`: with `allow_single_cluster=True` (needed so the *first* new defect
   type can form a cluster on its own), HDBSCAN turned 5 random unit vectors into one
   "cluster" in 64-d and 512-d (checked). (b) Merge of close compact parts + absorption of
   buffer points within `known_radius_factor` × radius: HDBSCAN labelled 9 of 15 points of one
   homogeneous blob as noise and split growing blobs, which would have raised a second alert
   for the same defect. Regression test: one blob fed point by point alerts exactly once
   (10 seeds).
2. **`known_radius_factor` 1.5, not 1.0**: with the radius a 95 % quantile of few points,
   fresh members of a named class fell outside it.
3. **Toy linkage uses the detector's training goods and no background defect** (config
   comment). With the provisional threshold, false alarms on a small repeated good-image
   pool would themselves form a cluster and open a bogus case; background spots would form a
   cluster before the fault. Both are real risks for the real demo (KNOWN_ISSUES #46) but
   would confound the plumbing test.
4. **No symptom-vocabulary change.** `new_visual_defect_cluster` is reported context only;
   adding it to `symptoms.yaml` would change the vocabulary version recorded by the frozen
   calibration run.
5. **Capability status is derived in `iqs_common.capabilities`** from the existing boolean
   flags and one table of reasons, so `iqs_vision/api/main.py` needed no change beyond the
   approved router lines (plus one flag line).
6. **`POST /v1/discovery/inspect` takes `inspected_at`** (part inspection time, simulated
   time for simulator parts), so cluster windows are in process time, not wall-clock time.
7. **Dashboard quality-loop client is a new module** (`loop_client.py`), not added to
   `client.py`, which the `step-2` branch may extend.
8. **Graph writes are configurable** (`IQS_DIAGNOSIS__GRAPH_WRITES=none|neo4j`, default
   none); with none the feedback reports `graph: disabled` instead of pretending.
9. **(7g) Workers live in an override file, not in the `apps` profile.** Without a case
   library the diagnosis worker exits and would restart forever, which would break
   `make up PROFILES=apps`. Gate-1 stack = `apps` profile + `deploy/compose/docker-compose.loop.yml`
   (`make up-loop`). The e2e test starts its own project `iqs-e2e` (throwaway credentials,
   own host ports 15433/17474/17687/18001/18002 via `IQS_*_HOST_PORT` in the base file, own volumes, removed afterwards) from base + loop +
   `docker-compose.e2e.yml`; migrations are applied from the host before the APIs start.
10. **(7g) Diagnosis image had no `iqs_mcp`** (pre-existing bug: the Dockerfile copied the
    common and diagnosis sources only). Invisible until now because the compose diagnosis
    service never had a case library, so the case service (which imports `iqs_mcp`) never
    started. Fixed by copying `packages/mcp_server/src`.
11. **(7g) Trigger bug found by the e2e test, fixed in `trigger.py`** (+2 regression tests,
    both fail on the old code). After naming, 2 of the 15 cluster members (distance 0.147
    and 0.175 from the frozen centroid) lay outside the known-class reach (1.5 × 0.095 =
    0.143). A grown cluster's radius had been computed over the core's reach only, not over all
    members. Those fringe scratches formed a second "new" cluster (centroid 0.145 from the
    named one, inside the same-type rule 0.229), so a second alert fired. Fixes: (a) a grown
    cluster's geometry is computed over all its members; (b) a group close to a **named**
    class (same rule as the merge) raises no alert and stays buffered (`near_named`, logged).
    The in-process test missed it because it did not stream parts while the case was waiting.
12. **(7g) e2e streaming starts 100 parts before the fault onset** (the harness reads
    ground truth for that; services never do). Earlier parts are training goods and route
    `normal` by construction. The stream pauses from the alert until the case awaits
    confirmation, then 30 more parts arrive (cluster grows, no second alert) before the
    redelivery check and the confirmation.
13. **(7g) Case library for the e2e** = the reference scenarios minus
    `drive_fault_m1` / `handling_error_shift_b` (as the unit agent fixtures): 8 cases with
    simulator ground-truth causes, built by the harness (no corpus on this machine). The
    scorer parameters come from the frozen calibration run `cal_dx_corpus_v2_dev_a010`,
    which was fitted on a different library (dx_corpus_v2 train), so the ranking quality in the
    e2e means nothing; the cause set is reported with `guarantee_applies: false` anyway.
14. **(7h-fix, #46) Distinct images, not observations, count toward `min_cluster_size`.**
    Identity = SHA-256 of the decoded pixels (mode, size, bytes), stored in the new nullable
    column `vision.discovery_observations.image_sha256` (migration **0006**, not an edit of
    0005, so a database already at 0005 stays valid). Repeats are one point for HDBSCAN, the
    size rule and the geometry; all their observations still become cluster members (the
    event `size` counts observations). The metal_nut linkage now lists its good images
    explicitly (`good_images`, checked against the step-2 split by
    `tests/test_image_linkage_config.py`) and deals each pool without replacement
    (`sampling: without_replacement`, one seeded permutation per cycle; the per-part draw of
    the main stream is kept, so the ADR-008 alignment rule holds). The toy category (unit
    and e2e tests) keeps its settings.


### Measured so far (in-process unit test, not a quality result)

`packages/diagnosis/tests/test_quality_loop_inprocess.py`, 2026-11-21, this laptop, under
`nice -n 19` while the `step-2` training ran. **Simulator data (tool_wear_m1, 72 h, seed 7),
synthetic image-to-part linkage, synthetic stripe images, random-init tiny PatchCore.**

| Quantity | Value |
| --- | --- |
| parts inspected after fault onset until the alert | 150 |
| anomalous images in the buffer at the alert | 5 (= `min_cluster_size`) |
| wall-clock, first anomalous image → case opened (incl. agent run) | 2.7–2.9 s (2 runs) |
| cases opened / alerts / confirmations | 1 / 1 / 1 |
| later scratch images routed to the named class | ≥ 90 % of ≥ 20 (assertion) |

### Measured in 7g: compose e2e gate 1 (plumbing, not a quality result)

`make test-e2e`, 2026-11-22 10:02 UTC, this laptop (WSL2, 16 vCPU, ~9.7 GiB VM), no other
stack running; report `reports/quality_loop/e2e_20261122T100245Z.json`. **Simulator data
(tool_wear_m1, seeds 7–11), synthetic image-to-part linkage, synthetic stripe images,
random-init tiny PatchCore, 8-case toy library.** 5 repeats, one per category `toy`…`toy5`,
all in one stack.

| Quantity | Per repeat (seed 7, 8, 9, 10, 11) | Median |
| --- | --- | --- |
| images inspected after fault onset until the alert | 150, 103, 149, 143, 117 | 143 |
| anomalous images at the alert (= cluster size) | 5, 5, 5, 5, 5 | 5 |
| wall-clock, first anomalous image → case listed by the API | 3.79, 2.59, 2.74, 3.75, 2.77 s | 2.77 s |
| wall-clock, first anomalous image → `awaiting_confirmation` | 5.13, 3.53, 3.56, 4.86, 3.78 s | 3.78 s |
| alert → case listed (worker poll 1 s + API poll 0.05 s) | 0.87, 0.62, 0.03, 0.33, 0.96 s | 0.62 s |
| cases opened / alerts / `case_confirmed` events | 1 / 1 / 1 each | — |
| false `known` routing before confirmation | 0 | — |
| defect images routed `known:scratch` after naming | 20/20 each | — |

Also asserted: redelivered event → still one case, one ticket, agent not re-run; count n → n+1
(0 → 1); one `CONFIRMED_CAUSE` edge, `INDICATES.count` = Postgres count; candidate-pool row
with `used_in_calibration_run = null`; calibration file sha256 `58eb7edb…` identical in the
container before/after and equal to the repo file; all 10 outbox rows `processed`, none failed.
Stack prepare 25 s (simulate, link, fit 5 tiny models, library), stack up 39 s, the 5
loops 86 s, test total 161 s.

Observed, not asserted: the conformal (APS) sets hold 8 of the 10 causes in every repeat
(the true cause is ranked first each time, but with probability 0.13: a nearly flat ranking), and novelty was flagged in 2 of 5 repeats. With
an 8-case toy library and scorer parameters from another library, this says nothing about
diagnosis quality.

Second full run after the compose port change (10:11 UTC, `e2e_20261122T101120Z.json`): passed;
median 2.66 s first anomalous → case, 3.59 s → `awaiting_confirmation`.

Integration: `make test-integration` 31 passed in 85 s (11 new in `test_quality_loop_db.py`).

### Measured in 7h: real-image demo (gate 3) and false-cluster risk (#46)

2026-11-22, this laptop, CPU inference (median 0.58–0.64 s per inspection with the
WideResNet-50 deploy model). **REAL MVTec AD metal_nut images (read-only symlinks, D2),
step-1 deploy PatchCore with the step-2 conformal threshold (α = 0.05, q̂ = 1.911); SIMULATED
process data; SYNTHETIC image-to-part linkage (good parts show `test/good`, tool-wear parts
`test/scratch`, sampled with replacement).** Symlinks: `models/patchcore`, `models/weights`,
`data/raw/mvtec_ad` (one level below the tracked placeholders, so nothing tracked changes;
git-ignored, not committed). A checksum + mtime listing of the main checkout's
`models/` and `data/raw/mvtec_ad/metal_nut` (670 entries) was identical before and after.

| Run | Report | Result |
| --- | --- | --- |
| false clusters: baseline (no fault), seed 7, all 2393 parts | `demo_false_clusters_20261122T104212Z.json` | 3/22 distinct good images above q̂; 335 anomalous parts (14 %); first alert after 45 parts; 1 cluster (335 members, radius 0.454); **1 bogus case** |
| demo: tool_wear_m1, seed 7, from 30 parts before onset | `demo_demo_20261122T110940Z.json` | alert at part 795 (5 after onset) from **6 anomalous good images, 0 defect images**; case → awaiting confirmation 2.2 s after the alert (24.0 s after the first anomalous image); top-3 TOOL_WEAR 0.132 / WRONG_PARAMETERS 0.129 / INSUFFICIENT_COOLING 0.127, APS set 8 of 10 causes, `guarantee_applies: false`; scientist: TOOL_WEAR supported, COOLING supported, PARAMETERS inconclusive; oracle confirmation TOOL_WEAR → count 0 → 1, graph written, cluster named "scratch"; afterwards 20/20 scratch parts and 27/244 good parts routed `known:scratch`, no new alert, 1 case in total |

Reading: the plumbing works on real images end to end, and the agent's ranking and scientist
point to the true cause, but they would have done that for this run whatever the trigger did.
The trigger itself fired on detector false alarms that repeat because the good pool is 22
images, and the named class is that false-alarm cluster. Gate 3 is therefore recorded as
**met for the plumbing only**. Options to fix are listed under KNOWN_ISSUES #46 (owner decision).

### Re-measured after the #46 fix (a) + (c), 2026-11-22

Same machine, model, threshold and labels as above; **good parts now show the 66 metal_nut
goods the deploy model never trained on (step-2 `calib` 44 + `test/good` 22), dealt without
replacement; the trigger counts distinct images.**

| Run | Report | Result |
| --- | --- | --- |
| false clusters: baseline, seed 7, all 2393 parts | `demo_false_clusters_20261122T173916Z.json` | **5/66** distinct goods above q̂ (test 007, 012, 016; calib 020 = 1.925 and 046, whose step-2 score equals q̂); 181 anomalous parts (7.6 %); first alert after 44 parts; 1 cluster of exactly those 5 images (181 members, radius 0.495 vs gate 0.5); **still 1 bogus case** |
| demo: tool_wear_m1, seed 7 | `demo_demo_20261122T180543Z.json` | alert at part 854 (64 after onset); 6 anomalous images before it: **5 good + 1 scratch**; case → awaiting confirmation 2.35 s after the alert; same top-3 (TOOL_WEAR 0.132 / WRONG_PARAMETERS 0.129 / INSUFFICIENT_COOLING 0.127), APS set 8 of 10, `guarantee_applies: false`; oracle TOOL_WEAR → count 0 → 1, graph written, named "scratch"; afterwards 20/20 scratch and 13/190 good parts `known:scratch`, no new alert, 1 case |

Reading: the fix removes the repeat effect (no cluster from fewer than 5 distinct images; the
anomalous share fell from 14 % to 7.6 %) but not the false cluster: with 66 goods, 5 exceed
q̂ (≈ 3.3 expected at α = 0.05), which is exactly `min_cluster_size`, and they are compact
enough (0.495) to pass the radius gate. One of the five is the calibration image that defines
q̂. Gate 3 stays **met for the plumbing only**; remaining options under KNOWN_ISSUES #46.

Owner decision (2026-11-22): no further trigger change in step 7. A margin, cluster size or
radius read off these runs would be tuned on the test data; the step-6 review band, or values
chosen on separate seeds and categories, go to step 8c. #46 stays open.

e2e re-run with the distinct-image trigger (`e2e_20261122T181241Z.json`, toy pool of 6 scratch
images per category, with replacement): **passed**, 5 repeats; median 3.41 s first anomalous →
case opened, 4.44 s → `awaiting_confirmation`; images after onset to alert 150, 127, 159, 155,
143; cluster size at alert (observations) 5, 7, 11, 9, 11, because repeats of the 6 images no
longer count toward the 5.

### Proposed KNOWN_ISSUES (to copy at merge; numbers #40+)

| # | Issue | Impact | Next action |
| --- | --- | --- | --- |
| 40 | Image ↔ part linkage is synthetic (`iqs_sim.linker`). | Vision-originated diagnoses show the loop works, not that images relate to process data. | Labelled in events, case text and dashboard. |
| 41 | Vision-triggered cases are not exchangeable with the simulator alarm cases of the calibration run. | Cause sets for them carry `guarantee_applies: false`. | A vision-source calibration run (needs ≥ 9 cases from a frozen scenario family). |
| 42 | Step 7 was built before steps 2 (conformal threshold) and 6 (cost decisions) merged. | Trigger uses the provisional max-of-calibration threshold. | Re-run the 7g gate after both merge. |
| 43 | Trigger parameters are untuned starting values (`configs/vision/discovery.yaml`). | No claim about detection delay or purity. | Step 8c evaluation. |
| 44 | Dirichlet counts are recorded, not used by the scorer. | "Probabilities update" holds only once 8a consumes them. | Step 8a. |
| 45 | Migration 0005 may collide with a 0005 on `step-2`. | Alembic chain breaks at merge. | Second branch to merge renumbers and fixes `down_revision`. |
| 46 | False alarms on good parts can form a cluster (repeated images, provisional threshold). | Bogus cases in the real demo. | Measure in 7h; then step 2 threshold or a minimum score margin. |
| 47 | The agent reads the whole run's observed data, also after the alert (replay, not live). | Same as for simulator alarm cases in steps 3–5. | Cut observed data at `alarm_ts` if live behaviour is to be shown. |

### Proposed ADRs (to copy at merge; numbers 015+)

- **ADR-015 — Quality-loop delivery:** transactional outbox in `platform.events`, one
  consumer per event type, at-least-once with idempotent handlers (`processed_events`,
  case ID = UUIDv5(cluster), unique `source_cluster_id`, feedback once per case); event
  schema v1 extended with optional fields; pinned contract files; PatchCore patch features
  instead of DINOv2 (D1); blueprint "core" profile = our `apps` profile.
- **ADR-016 — Feedback policy:** counts per (defect type | source, cause), candidate pool
  only, frozen calibration and library (D3); Neo4j subset Case/Cause/DefectType with MERGE
  and Postgres as the source of truth for counts; no-listed-cause confirmations record
  nothing.
