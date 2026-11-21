# Step 3 — diagnosis agent (minimum version): notes for merging

Branch `step-3` (from `step-5`, worktree `~/code/iqs-step5`), started 2026-11-20. Like
`docs/notes/step-5a.md`, shared docs are not edited on this branch; this file collects
what goes into them at merge. `step-1` is not merged here (it lands on `main` by PR first).

> **Merged 2026-11-21.** The text meant for the shared docs is now in CHANGELOG, KNOWN_ISSUES,
> TECHNICAL_DECISIONS (ADR-006), TESTING, EXPERIMENTS, ARCHITECTURE and IMPLEMENTATION_PLAN.
> This file stays as the detailed record.

Minimum version = 3a–3d on simulator cases only: no LLM, no model downloads, no new
dependencies, no paid calls. LLM gateway (Ollama only, no paid API before step 8),
NHTSA/AI4I, dense retrieval + reranker, Neo4j and the dashboard follow in 3e–3g.

---

## 3a — simulator case corpus and symptom profiles

| Module / file | Role |
| --- | --- |
| `iqs_diagnosis/domain/case.py` | `DiagnosisCase` (no label field), `SymptomObservation`, `SymptomProfile` |
| `iqs_diagnosis/data/process_store.py` | Read-only DuckDB over `observed/` (allow-list → external access off → lock) |
| `iqs_diagnosis/ingestion/simulator.py` | `load_case(run_dir)` from `cases/case.json` + `observed/`; no `iqs_sim` import |
| `iqs_diagnosis/symptoms/simulator_profile.py` | 19 descriptive features, reference period (first 12 h) vs affected period (from alarm window start) |
| `iqs_diagnosis/symptoms/vocabulary.py` + `configs/diagnosis/symptoms.yaml` | 19 symptom codes with feature rules |
| `iqs_diagnosis/evaluation/labels.py` | The only reader of `ground_truth/case_labels.json` |
| `iqs_diagnosis/evaluation/corpus.py` + `configs/diagnosis/sim_corpus.yaml` | Splits, profiles/labels JSONL (separate files) |
| `iqs_sim.pipeline.run_family` + `simulate.py run-family` | Runs a family into opaque `run-<sha12>/` dirs + `runs_index.json` |
| `scripts/build_sim_corpus.py` | family → runs → profiles + labels; manifest `eval_sets/diagnosis/sim/dx_corpus_v1.json` |

### Symptom vocabulary (v0.1.0)

Thresholds were set by looking at the 11 reference scenarios only (never at corpus
splits). Retrieval also uses the numeric features, so a threshold only decides the
readable code.

| Code | Rule | Seen in reference scenario |
| --- | --- | --- |
| `reject_rate_increase` | from the case (alarm) | all |
| `reject_reason_dimension` | dimension share of rejects ≥ 0.5 | fixture, gauge drift, handling |
| `reject_reason_surface` | surface share ≥ 0.5 | the 7 process/material/environment faults |
| `rejects_concentrate_machine` (+where) | S2 machine reject lift ≥ log2 0.5 (×1.4) | fixture M2B, overload M2A |
| `rejects_concentrate_batch` (+where) | batch lift ≥ 1.5 (×2.8) | bad batch B0008 |
| `rejects_concentrate_shift` (+where) | shift lift ≥ 1.0 (×2) | handling shift B |
| `dimension_mean_shift` | \|Δ mean dimension\| ≥ 1 SD | fixture, gauge drift |
| `dimension_spread_increase` | SD ratio ≥ ×1.4 | fixture, handling |
| `dimension_shift_one_machine` (+where) | M2A vs M2B shift gap ≥ 1.5 SD | fixture M2B |
| `gauge_check_drift` | master-part check moved ≥ 6 µm | gauge drift |
| `machine_temperature_rise` (+where) | max machine temp z ≥ 2 and min < 2 | coolant M3, overload M2A |
| `temperature_rise_all_machines` | min machine temp z ≥ 2 | ambient |
| `ambient_temperature_high` | ambient +2 °C vs reference | ambient |
| `spindle_load_rise` (+where) | load z ≥ 1.5 | overload, tool wear, wrong parameters |
| `batch_load_rise` (+where) | one batch's load z ≥ 2.5 | bad batch |
| `vibration_rise` (+where) | vibration z ≥ 2 | tool wear |
| `tool_wear_rate_rise` (+where) | wear per cut ≥ ×1.19 | tool wear |
| `power_deviation` (+where) | actual/commanded residual z ≥ 3 | drive fault |
| `setpoint_change` (+where) | ≥ 5 % cuts with new parameter set / speed outside window | wrong parameters |

### Corpus `dx_corpus_v1`

Built 2026-11-20 with `nice -n 19 uv run python scripts/build_sim_corpus.py --workers 2`
while a PatchCore training job held the CPU (16 logical cores, load ≈ 5).
Manifest: `eval_sets/diagnosis/sim/dx_corpus_v1.json` (family, profile and label hashes).

| Split | Family / seed | Scenarios | Cases (alarm raised) | Simulate | Profiles |
| --- | --- | --- | --- | --- | --- |
| train (library, 8 causes) | dxtrain_v1 / 31001 | 240 | 225 | 158 s | 69 s |
| dev (10 causes) | dxdev_v1 / 31002 | 100 | 94 | 70 s | 32 s |
| test (10 causes) | dxtest_v1 / 31003 | 200 | 188 | 144 s | 63 s |

Total 537 s inside the script, 8 min 58 s wall clock, 170 % CPU, peak RSS 224 MB;
442 MB on disk under `~/iqs-data/sim/dx_corpus_v1`. 33 of 540 runs raised no alarm and
therefore no case (mostly `spindle_overload` and `bad_material_batch` at small
magnitudes), consistent with the 5a detection numbers.

**Manifest:** `eval_sets/diagnosis/sim/dx_corpus_v1.json` (family, profile and
label SHA-256 values), committed after the user approved excluding
`eval_sets/diagnosis/sim/*.json` from detect-secrets (same convention as step 1's
`eval_sets/vision/*`; `test_corpus_manifest.py` validates the schema instead). Merge note:
step 1 edits the same `exclude:` line in `.pre-commit-config.yaml`; combine both paths.

---

## 3b — retrieval, scoring, novelty (pure domain code)

| Module / file | Role |
| --- | --- |
| `retrieval/bm25.py` | BM25Okapi over the rendered symptom text (`code` and `code@where` tokens) |
| `retrieval/vector.py` | asinh-compressed, library-standardised features; similarity 1/(1 + RMS distance) |
| `retrieval/fusion.py` | Reciprocal-rank fusion, k = 60 |
| `retrieval/hybrid.py` | BM25 + vector → RRF, plus the max vector similarity for novelty |
| `scoring/votes.py` | RRF-weighted votes of the top n neighbours, additive smoothing over all 10 causes; grid fit by dev NLL |
| `novelty/threshold.py` | Order-statistic threshold on dev known-case similarities: P(known case flagged) ≤ α under exchangeability |
| `scoring/ranker.py` | `CauseRanker` (top 3 + probabilities + novelty + neighbours), `fit_ranker`, `RankerParams` JSON |
| `scripts/fit_ranker.py` → `configs/diagnosis/ranker_params.json` | Fit on train (library) + dev; test split not read |

**Dev fit (tuning data, optimistic by construction; not the 3d evaluation):** library
225 cases / 8 causes; chosen n_neighbours = 5, alpha = 0.001 (the smallest grid value);
novelty α = 0.05 → threshold 0.708 on 74 known dev cases. Dev top-1 74/74, top-3 74/74,
NLL 0.012; false novelty 2/74 (0.027); novel cases detected 20/20 (drive fault and
handling error, absent from the library). Fit takes under 2 s.

Points for later steps:
- alpha at the grid edge means very peaked probabilities: dev gives no evidence against
  overconfidence because dev has no errors. Step 4 (conformal sets) must calibrate on
  data with errors; a harder corpus (smaller magnitudes, two-fault cases) would be needed
  for informative sets.
- Simulator faults leave clean, distinct signatures; near-perfect simulator accuracy is
  expected and must not be presented as real-world accuracy.

---

## 3c — agent graph, 8D draft, MCP ticket tools, API

| Module / file | Role |
| --- | --- |
| `agent/graph.py` | LangGraph: intake → extract_symptoms → retrieve → novelty_check → score_causes → draft_8d → open_ticket (MCP) → confirm (**interrupt**) → close_ticket (MCP). JSON-only state; tool calls recorded in `tool_calls`; per-node timings |
| `agent/tickets.py` | `TicketGateway` port; `McpTicketGateway` (MCP client: in-process server, stdio or URL) |
| `agent/checkpoint.py` | `InMemorySaver` (tests, eval) / `PostgresSaver` (service; case pauses survive restarts) |
| `reporting/eight_d.py` | D0–D8 schema; template draft of D0/D2/D4; D1, D3, D5–D8 `pending_engineer` and must be empty; completeness = agent sections valid / 3 |
| `persistence/repository.py` | `CaseRepository` port: in memory + SQL (`diagnosis.cases`, `case_causes`, `reports_8d`) |
| `iqs_mcp/{models,store,server}.py` | Mock QMS: `create_ticket` (idempotent per open case), `get_ticket`, `update_ticket`; in-memory + SQL (`qms.tickets`, `ticket_events`); `python -m iqs_mcp` (stdio) |
| `migrations/versions/0002_…` | The five tables (rendered offline with `alembic upgrade 0001:0002 --sql`; checked against Postgres by the integration test tomorrow) |
| `api/cases.py`, `api/wiring.py`, `settings.py` | `POST /v1/cases` (run_id under `IQS_DIAGNOSIS__RUNS_ROOT`), `/run`, `GET /{id}`, `/cause-set` (kind `top3` until step 4), `/report-8d` (en only before 3e), `/confirm`; features `cases`, `retrieval`, `report_8d` switch on when `IQS_DIAGNOSIS__LIBRARY_PATH` is set |
| `retrieval/library.py` + `evaluation.corpus.write_library` | The agent reads a library file of confirmed past cases, never a labels file |

Design points:
- `/run` is synchronous up to the confirmation interrupt: without an LLM a run is under a
  second (measured in 3d). With LLM nodes (3e) it becomes 202 + job ID.
- Tool calls run in their own nodes *after* the interrupt, because LangGraph re-runs an
  interrupted node from its start.
- A failed QMS call is recorded (`ok: false`) and the case still reaches the engineer.
- Ticket severity rule (stated, not learned): reject rate after the alarm ≥ 15 % → 1,
  ≥ 7 % → 2, else 3.
- Causes are stored by taxonomy code; the `cause_taxonomy` table of §2.4 is not created yet.

**Integration tests written, not run (no Docker tonight):**
`tests/integration/test_diagnosis_persistence.py` (migration 0002 tables, SqlTicketStore,
case survives a restart with the Postgres checkpointer). Run with `make test-integration`.

**Dependencies (approved 2026-11-20):** `mcp` and `iqs-mcp` (workspace source) added to
`packages/diagnosis/pyproject.toml`; `uv.lock` gained 4 lines in the `iqs-diagnosis`
entry only (resolved offline, no version changes, nothing downloaded).

---

## 3d — evaluation (simulator corpus, test split, run once)

> **Simulator accuracy is not real-world accuracy.** The faults are injected into a
> line model whose signals were designed so that each fault is observable and
> distinguishable (step 5a). These numbers show that the pipeline works end to end
> and that retrieval separates the designed signatures; they say nothing about
> NHTSA cases or a real plant. Real-data results (3f) will be reported separately.

`scripts/evaluate_diagnosis.py` runs the **full agent** per case (symptoms from
`observed/` → retrieval → novelty → votes → 8D → `create_ticket` over MCP → simulated
engineer confirms the true cause → `update_ticket`). Library = train split (225 cases,
8 causes); parameters fitted on dev (3b) and frozen; the test split was evaluated
**once** (2026-11-20, `nice -n 19`, 68 s for 188 cases). Report:
`reports/diagnosis/dx_corpus_v1_test.json` (dev run alongside, tuning data).

**Test split: 188 cases, 148 with a library cause, 40 with a held-out cause**
(95 % Wilson intervals):

| Metric | Result |
| --- | --- |
| Top-1 accuracy, known causes | **1.000** [0.975, 1.000] (148/148) |
| Top-3 accuracy, known causes | 1.000 [0.975, 1.000] (148/148) |
| Top-3 accuracy, all cases (held-out causes cannot be named) | 0.787 [0.723, 0.840] (148/188) |
| Novelty detected (held-out causes flagged "no close past case") | **1.000** [0.912, 1.000] (40/40) |
| False novelty (known causes flagged as new; target α = 0.05) | **0.061** [0.032, 0.112] (9/148) |
| Tool-call accuracy (create + update, arguments and QMS state checked) | 1.000 [0.990, 1.000] (376/376) |
| 8D completeness (D0/D2/D4 valid) | 1.000; 188/188 complete |
| Latency to the confirmation point | p50 317 ms, p95 414 ms, max 564 ms |
| LLM cost | 0 (no LLM in this version) |

| Baseline (known causes) | Top-1 | Top-3 |
| --- | --- | --- |
| Random over 10 causes (expected) | 0.100 | 0.300 |
| Majority (most frequent library causes) | 0.135 [0.089, 0.200] | 0.405 [0.330, 0.486] |
| BM25 on symptom codes only | 0.973 [0.933, 0.989] | 0.993 [0.963, 0.999] |
| Vector on features only | 1.000 [0.975, 1.000] | 1.000 [0.975, 1.000] |
| **Hybrid (agent)** | **1.000** [0.975, 1.000] | **1.000** [0.975, 1.000] |

Per fault, top-1 is 1.000 for all eight library causes (15–20 cases each) and 0 for the
two held-out causes by construction.

Reading the results honestly:
- **The task is easy by design.** The feature vector alone is perfect; BM25 on the
  readable codes loses 4 cases (ambient ↔ bad batch, coolant → bad batch). The hybrid
  cannot beat a perfect component here, so this corpus cannot show the value of fusion.
- **False novelty 6.1 % exceeds the 5 % target.** The order-statistic bound holds on
  average over calibration sets; one dev set of 74 known cases (k = 3, bound 3/75 = 4 %)
  gives a threshold whose conditional rate scatters around that, and the interval
  [3.2 %, 11.2 %] includes 5 %. The 9 cases spread over 6 faults (similarity 0.52–0.70,
  threshold 0.708). On test, held-out cases reach at most 0.528 similarity, so a lower
  threshold would separate almost perfectly, but choosing it from test would be test
  tuning; it was not changed. Step 4 should calibrate on a larger dev set.
- Only the agent sections of the 8D are measured for completeness; D1, D3, D5–D8 are
  left to engineers on purpose.
- Tool-call accuracy checks arguments against the agent's own state and the QMS ticket
  afterwards: it shows that the MCP round trip is faithful, not that an LLM chose tools
  well (there is no LLM yet).

Deterministic CI subset: `packages/diagnosis/tests/test_eval_harness.py` runs the
same harness on the 10 reference cases with every `pytest` run (no LLM, no Docker,
about 25 s). No separate `eval.yml` workflow was added: the existing CI test job runs it.
Results are written as JSON reports (MLflow is a vision-package dependency and was
not added to diagnosis).

### Step 4 needs a harder family

The conformal step needs probability vectors that are sometimes wrong. On
`dx_corpus_v1` the agent makes no top-1 errors on known causes and the fitted vote
smoothing sits at the grid edge (alpha = 0.001), so conformal sets would be trivial
(size 1) and coverage uninformative. Before step 4, build a harder family:
- **smaller magnitudes** near each fault's detection limit (for example the lower
  half of each sampling range, or below it), so signatures overlap with noise;
- **two-fault cases** (the engine supports several faults per scenario, 5a open
  question 6), labelled with the primary cause by a written rule, or evaluated with a
  set-valued truth;
- optionally fewer library cases per cause and faults on different machines than in
  the library;
- a larger dev split (for example 30 per cause) so the novelty and conformal
  thresholds are calibrated on more than 74 cases.

---

## dx_corpus_v2 — harder corpus for step 4

> **Commit numbering note.** This work was committed as `a55bb36 feat(step-3e): …`. In the
> step-3 plan, **3e is the LLM gateway (Ollama only)**, which is not built yet. The commit
> was not renamed (user decision, 2026-11-20); the corpus belongs to the step 3→4 hand-over.
> When the LLM gateway is built it keeps the label 3e in docs, and this commit should be
> read as "step 4 preparation". (built 2026-11-20; test split not evaluated)

Config `configs/diagnosis/sim_corpus_v2.yaml`; manifest `eval_sets/diagnosis/sim/dx_corpus_v2.json`.

- **Near-limit magnitudes** (single-fault scenarios): `iqs_sim.scenarios.near_limit_range` =
  `[floor + (lo − floor)/2, lo + (hi − lo)/4]` (floor = `min_abs_magnitude`, [lo, hi] = v1
  sampling range). One rule for every fault. It was chosen from a 160-run detection-curve
  measurement (16 runs per fault across [floor, lo]; alarms only, no diagnosis results):
  tool wear, overload, coolant and bad batch barely alarm below lo; setpoint change
  and handling alarm even at the floor; gauge drift, fixture, ambient and drive switch on
  mid-way. The first try ([floor, lo]) gave almost no bad-batch or tool-wear cases.
- **Two-fault scenarios** (standard magnitudes, two distinct fault types), primary cause by
  the excess-defects rule (labelling guide, simulator section; `evaluation/labels.py`).
  All 280 pair scenarios raised an alarm.
- **Larger dev split**: at least 32 cases per cause (32–82).

| Split | Single / pair scenarios | Cases (two-fault) | Simulate | Profiles |
| --- | --- | --- | --- | --- |
| train (8 causes) | 320 / 80 | 306 (80) | 168 s | 51 s |
| dev | 600 / 120 | 582 (120) | 304 s | 117 s |
| test | 600 / 120 | 582 (120) | 394 s | 98 s |

Build: 1133 s in the script, **18 min 54 s wall clock** (`nice -n 19`, 2 workers, 177 % CPU,
peak RSS 387 MB, PatchCore training running), 1.5 GB on disk.

**Dev results of the frozen v1 method** (v1 vocabulary, v1 ranker parameters:
5 neighbours, alpha 0.001, novelty threshold 0.708; nothing refitted). Report
`reports/diagnosis/dx_corpus_v2_dev_v1method_v2library.json`. 582 cases: 449 with a library
primary cause (349 single, 100 two-fault), 133 with a held-out primary cause.

| Metric (dev, 95 % Wilson) | v2 train library | v1 train library |
| --- | --- | --- |
| Top-1, known causes | 0.927 [0.899, 0.947] | 0.933 [0.906, 0.953] |
| Top-3, known causes | 0.982 [0.965, 0.991] | 0.949 [0.924, 0.966] |
| Top-1, single-fault (near limit) | 0.991 (346/349) | 0.997 (348/349) |
| Two-fault: top-1 = primary | 0.700 [0.604, 0.781] | 0.710 |
| Two-fault: primary in top-3 | 0.930 [0.863, 0.966] | 0.780 |
| Two-fault: both causes in top-3 | 0.540 [0.443, 0.634] | 0.240 |
| Two-fault: top-1 is one of the injected causes | 0.960 | 1.000 |
| Novelty detected | 1.000 (133/133) | 1.000 |
| False novelty (target 0.05) | **0.263** [0.224, 0.305] | **0.401** |
| BM25-only / vector-only top-1 (known) | 0.898 / 0.911 | 0.873 / 0.918 |

Reading:
- v2 is harder where it matters for step 4: the method now makes errors (33/449 top-1),
  mostly on two-fault cases, where the second cause often wins top-1 (top-1 is one of
  the injected causes 96 % of the time, but the primary only 70 %).
- On v2 the hybrid beats both of its parts (0.927 vs 0.898 / 0.911), which v1 could not show.
- Near-limit single faults are still easy for the feature vector (0.991). Weak faults
  mostly change whether a case exists (alarm yield), not how separable it is once it does.
- The v1 novelty threshold does not transfer: 26 % false novelty with the v2 library.
  It was calibrated on standard-magnitude cases; near-limit and two-fault cases lie
  further from the library. Recalibrating on v2 dev is a step 4 task.
- The pipeline found a bug in the harness on the first dev run (pair runs in
  `runs/<split>-pairs` were not collected; fixed before these numbers).

---

## ADR

Moved to `docs/TECHNICAL_DECISIONS.md` as **ADR-006** at the merge into `main` (2026-11-21).
