# Step 3 — diagnosis agent (minimum version): notes for merging

Branch `step-3` (from `step-5`, worktree `~/code/iqs-step5`), started 2026-11-20. Like
`docs/notes/step-5a.md`, shared docs are not edited on this branch; this file collects
what goes into them at merge. `step-1` is not merged here (it lands on `main` by PR first).

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

**Not committed yet:** `eval_sets/diagnosis/sim/dx_corpus_v1.json` (family, profile and
label SHA-256 values). The detect-secrets hook flags the hashes as high-entropy strings;
excluding the path (as step 1 does for `eval_sets/vision/*`) needs the user's decision.

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

## Proposed ADR-006 (number may need renumbering at merge: step 1 holds ADR-004/005, step 5a has an unnumbered proposal)

### ADR-006 — Diagnosis minimum version: simulator symptom profiles, retrieval votes, novelty threshold (step 3a–3b)

**Status:** proposed · 2026-11-20

**Context.** Blueprint §5 Step 3 asks for a working agent with retrieval, novelty and
top-3 causes, measured on a test split. The real-data path (NHTSA, AI4I, LLM) needs
downloads, hand labels and an LLM; the simulator (step 5a) already gives labelled cases
with exact ground truth. Simulator case text is templated and does not say where
rejects concentrate, so text alone cannot separate the 10 causes.

**Decisions.**

1. **Minimum version without an LLM** (approved 2026-11-20). Probabilities come only
   from retrieval votes (§2.3 rule). The LLM arrives in 3e via Ollama only; no paid API
   before step 8 (user rule).
2. **Symptom extraction may compute descriptive concentration and shift statistics**
   (approved, answers 5a open question 2): effect sizes only, no p-values or verdicts;
   statistical tests with BH correction stay in the scientist agent (5b).
3. **Symptom codes are observations in lower snake case**, separate from cause codes
   (answers 5a open question 1). Rules live in `configs/diagnosis/symptoms.yaml`.
4. **Features are machine-agnostic** (max/min over machines, batches, shifts); the
   location is evidence (`where`). A tool-wear case on M1 and on M3 then look alike.
5. **Leak guards.** Only `iqs_diagnosis.evaluation` may name the ground-truth directory
   or be imported for labels (AST test); profiles and labels are separate files joined
   by `case_id`; run directories are opaque hashes; the process store is the locked
   DuckDB allow-list from 5a.
6. **Splits = separate scenario families** with their own names and seeds (train =
   case library, dev = tuning, test = reported once in 3d). Two causes
   (`MACHINE.DRIVE_FAULT`, `PERSONNEL.PART_HANDLING_ERROR`) are left out of the
   library to measure novelty; chosen before any results.
7. **Generated data outside the repo** (`~/iqs-data/sim/<corpus>`, answers 5a open
   question 3); only the manifest (family hashes, file hashes, counts, timings) and the
   fitted ranker parameters are committed.
8. **Dense embeddings and reranker deferred to 3f** (approved): for numeric simulator
   profiles a standardised feature vector is the "dense" side of hybrid retrieval.

**Consequences.**
- Simulator results say nothing about real-world accuracy; they are reported separately
  from NHTSA results (§6.3, §6.4).
- The vocabulary thresholds were read off the same reference scenarios the signature
  test uses; the test is a regression check, not evidence of accuracy.
- Changing a feature or rule changes every profile; the corpus manifest records the
  vocabulary hash, so stale profiles are detectable.
