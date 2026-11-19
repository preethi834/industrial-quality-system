# Industrial AI Engineering System — Technical Blueprint

Nov 15, 2026 · Preethi Mannaru

> **Revision 2 — 15 Nov 2026, after external review.** Corrections: independent uv projects with path dependencies instead of one workspace (§2.10, §4); full conformal mathematics, assumptions and shift limits (§2.8); decision layer constrained so the false-alarm guarantee survives (§2.5 P2-C); feature modularity so the integrated system runs without advanced features installed (§2.9); Windows/WSL2 + cloud-GPU workflow (§3.8); approval gates for downloads, paid APIs and GPU jobs (§3.9); measurable completion gates for every milestone (§5); updated checklist (§8).

> **Implementation note (19 Nov 2026, after Step 0).** Where this blueprint and the repository differ, the repository's ADRs in `docs/TECHNICAL_DECISIONS.md` win. In particular: the repo keeps **one uv workspace** (decided in Step 0a; isolation is achieved with `uv sync --package`, per-service Docker images and optional extras), so §2.10 and the per-project lockfiles in §3.3/§4 are **superseded**; compose profiles are `core` and `apps` (ADR-002); Postgres is on host port 5433 (ADR-001). §2.8 (conformal maths), §2.9 (modularity), §3.9 (approval gates) and the §5 step gates remain in force.

## 0. Executive summary

**What we are building:** one monorepo containing two connected systems plus a factory simulator that ties them together:

- **Project 2 — Open-World Defect Discovery (vision):** detect and localise defects, guarantee a maximum false-alarm rate, make cost-aware accept / reject / human-review decisions, discover new defect types, learn them from 5–10 examples, explain them (heatmaps, text, “repaired” image) and run on edge hardware.
- **Project 1 — Active Root-Cause Diagnosis Agent (GenAI):** retrieve similar past cases (hybrid RAG + Neo4j knowledge graph), admit when a problem is new, return a conformal set of causes, ask the most informative question, test hypotheses against process data, write the 8D report and propose FMEA updates.
- **Factory simulator (SimPy):** a small production line with injected, known root causes — the ground truth that makes root-cause accuracy measurable.
- **Integration:** a new defect cluster in Project 2 automatically opens a case in Project 1; the engineer confirms the cause; FMEA, knowledge graph and Bayesian network probabilities update.

**Guiding principles taken from the spec (p.7):** each layer is a finished, demonstrable result; “a finished base plus two advanced layers beats an unfinished project with everything”; every claim is measured; cost savings only with stated assumptions.

**How requirements are classified in this blueprint**

| Tier | Meaning | Contents |
| --- | --- | --- |
| **A — Core (mandatory)** | Roadmap steps 1–7. Must work end-to-end before anything else. | P2 Layers 1–3, P1 Layers 1–3, factory simulator + scientist agent, cost-aware decisions, integrated loop |
| **B — Advanced (required by spec, built one at a time)** | Roadmap step 8, in a fixed priority order | Smart questions, Living FMEA + feedback loop, defect discovery, few-shot + continual learning, edge INT8, own small LoRA model, “how it should look” counterfactuals |
| **C — Optional** | Marked optional in the spec, or infrastructure that adds no core capability | Diffusion-generated synthetic defects, OpenVINO (in addition to ONNX Runtime), GPU benchmarks if no GPU is available, Hugging Face Spaces demo, Langfuse self-hosting, MVTec leaderboard submission |
| **D — Release (step 9)** | Public proof of work | Simulator + 8D benchmark on Hugging Face, 4–6 page technical report, 2-minute demo video |

**Three key design decisions up front** (justified in later sections):

1. **Independently installable services** — the vision stack (PyTorch, Anomalib, OpenVINO) and the agent stack (LangGraph, LLM SDKs, pgmpy) are heavy and may conflict. Each deployable component is its own uv project with its own lockfile and virtual environment; a small `common` package is shared as a path dependency. *(Revision 2: corrected from a single uv workspace, which would force one shared resolution — see §2.10.)*
2. **PostgreSQL + pgvector as the single relational and vector store** (the spec allows Qdrant or pgvector). We need Postgres anyway for cases, inspections, FMEA proposals and the mock QMS/MES, so pgvector removes one service. Neo4j stays for the knowledge graph (spec requirement); DuckDB reads simulator Parquet files for analytics.
3. **A fixed root-cause taxonomy** — conformal prediction and Bayesian networks need a finite label space. Every cause the agent can name maps to a taxonomy ID; free text is used for explanation, never as the label.

## 1. Specification analysis (Task 1)

Source: *Two Advanced AI Engineering Projects* (8 pages). Everything below is extracted from the spec; where I add an interpretation it is marked **\[interpretation\]**.

### 1.1 Theme and differentiators (p.1)

Theme: *AI a factory can trust — it proves its answers, tests its own ideas, and knows what mistakes cost.* The spec states the topic itself is not unique (MVTec anomaly detection, AnomalyGPT, LLM+KG for FMEA already exist); uniqueness must come from the hard parts:

| Common project | This project must deliver |
| --- | --- |
| MVTec AD only | MVTec AD 2 (2025) + discovery of new defect types |
| “95% accurate” | Conformal statistical guarantees |
| Chatbot | Agent chooses the most informative next question |
| Static system | Learns from each confirmed case; FMEA kept current |
| Notebook | Dockerised, CI-tested, quantised, deployed, monitored |
| Text-only answers | Hypotheses tested against process data |
| No ground truth | Own simulator with injected causes |
| Cloud LLM only | Own fine-tuned small local model, compared on quality and cost |
| Accuracy only | Cost-aware accept / reject / ask-a-human in euros |
| Heatmap only | “Repaired” version of the part shown next to the defective one |
| Public data only | Releases own simulator and benchmark |

**The connected loop:** camera finds new defect cluster (P2) → case opens automatically (P1) → agent asks smart questions, returns guaranteed cause set → engineer confirms → FMEA updated, both systems learn.

### 1.2 Project 1 — Active Root-Cause Diagnosis Agent (p.2–4)

**Problem:** deviation / complaint knowledge (past cases, FMEA, machine and process data) is scattered; assistants give one confident answer even when guessing; they fail on never-seen problems; FMEAs go stale.

**Goal:** diagnose like an experienced engineer — reuse past knowledge, ask the most useful follow-up questions, return a set of likely causes with a proven reliability guarantee, write the 8D report, propose FMEA updates for undocumented failure modes.

| Layer | Feature | Key techniques |
| --- | --- | --- |
| 1 Base | Retrieve similar past cases (hybrid RAG + KG), suggest root causes, draft 8D report, open/update ticket via MCP tools, evaluation suite | RAG, agents, MCP, KG, LLM evaluation |
| 2 Novelty | Detect when no close past case exists and say so; top-3 ranked causes | Similarity thresholds, honest uncertainty |
| 3 Guarantees | Conformal prediction: “true cause in this set 90% of the time”, verified on held-out cases | Conformal prediction, calibration |
| 4 Smart questions | Bayesian network of causes and symptoms; next question = highest expected information gain; re-rank causes | Bayesian networks, entropy, agent planning |
| 5 Living FMEA | Compare confirmed cause with FMEA; propose new failure-mode entries for engineer approval | Quality processes, structured KG updates |
| 6 Feedback loop | Every confirmed case updates graph and probabilities | Human-in-the-loop, continual improvement |

**Signature moves:** **A — Scientist agent** (propose cause → query/statistical test on machine data → keep or reject → show evidence). **B — Factory simulator** (machines, sensors, material batches, injected known problems → exact accuracy). **C — Own small model** (LoRA fine-tune of a 1–8B open model on 8D/diagnosis data, run locally, compare quality and cost with the large model).

**Tech stack (p.3):** Python · LangGraph · MCP server for a mock quality/MES system · one API LLM (OpenAI / Anthropic / Azure OpenAI) + one local model via Ollama · hybrid RAG: Qdrant or pgvector + BM25 + reranker (e.g. bge-reranker) · Neo4j (parts, components, processes, failure modes, causes, FMEA entries) · pgmpy + NumPy/SciPy · conformal prediction (MAPIE or own) · DuckDB/SQL + SciPy + statsmodels · SimPy + NumPy/pandas with configurable faults · Transformers + PEFT (LoRA) / TRL or Unsloth on Qwen or Llama 1–8B · FastAPI + Streamlit (chat, cause set, evidence, 8D report, FMEA proposals) · Ragas or DeepEval + own labelled set; LLM-as-judge checked against human labels · Langfuse · Docker, docker-compose, GitHub Actions running evaluation on every push.

**Datasets:**

| Dataset | Type | Use |
| --- | --- | --- |
| NHTSA Recalls flat file (data.gov) | Real | Past cases: component, defect, consequence, remedy → knowledge base + graph |
| NHTSA Complaints flat file (data.gov) | Real | New incoming problems to test diagnosis |
| AI4I 2020 Predictive Maintenance (UCI) | Real (itself synthetic-derived, published) | Machine settings + failure types → Bayesian network and graph |
| Generated 8D reports + FMEA (DE + EN) | LLM-generated from recalls | Knowledge base content; must be labelled as generated |
| Simulated factory data | Synthetic (own simulator, released on HF) | Ground truth for evaluation + hypothesis testing |
| Hand-labelled test set, \~50–100 cases | Own labels | Part held out as “new problems”; calibration set for conformal |

**Metrics (p.3–4):** top-1 / top-3 accuracy (overall + held-out new problems) · conformal coverage (e.g. 90% target) and average set size · novelty detection rate (“no close match”) · questions to reach right cause: smart vs random · FMEA proposal precision (human-accepted) · simulator accuracy per injected fault type · hypothesis testing: share of correct causes confirmed and wrong causes rejected · small vs large model: quality, cost, latency · tool-call accuracy, 8D completeness, cost and latency per case.

### 1.3 Project 2 — Open-World Defect Discovery (p.4–6)

**Problem:** rare defects (few training examples); new defect types appear when materials/tools/processes change; models only know trained types, raise false alarms, give no reason, offer no error guarantee.

**Goal:** detect and localise, explain, notice a new defect kind, learn it from few examples without forgetting, guarantee a maximum false-alarm rate, run fast on edge hardware.

| Layer | Feature | Key techniques |
| --- | --- | --- |
| 1 Base | PatchCore / EfficientAD + supervised baseline; heatmaps; VLM description; FastAPI + Docker | PyTorch, anomaly detection, transfer learning |
| 2 Harder data | Evaluate on MVTec AD 2 and VisA, not only MVTec AD | Robust evaluation |
| 3 Guarantees | Conformal threshold: “at most 2% of good parts rejected”, verified on test data | Conformal anomaly detection |
| 4 Discovery | Embed unknown anomalies, cluster, alert when a new cluster grows, ask for a few labels | Embeddings, HDBSCAN, human-in-the-loop |
| 5 Few-shot + continual | Learn new type from 5–10 examples without forgetting | Prototypical networks, continual learning |
| 6 Edge | INT8 with ONNX Runtime / OpenVINO; accuracy + speed before vs after | Model optimisation, deployment |
| Optional | Synthetic defects via diffusion | Diffusion, augmentation |

**Signature moves:** **A — “How it should look”** (diffusion inpainting of the defect region → repaired image next to original). **B — Thinks in euros** (accept / reject / human; thresholds minimise expected cost with stated example costs).

**Tech stack (p.5):** PyTorch, torchvision, timm · Anomalib (PatchCore, EfficientAD) · fine-tuned ResNet/ViT; prototypical networks on DINOv2 embeddings · UMAP + HDBSCAN · replay buffer / prototype memory · anomaly maps, Grad-CAM, VLM (Qwen-VL or API) · conformal threshold calibration · cost-sensitive decision layer (NumPy, own) · diffusers (SD inpainting) · MLflow or W&B · ONNX, ONNX Runtime, OpenVINO INT8 · FastAPI + Docker, Gradio or Streamlit (labelling view for new clusters) · Evidently AI · GitHub Actions, Hugging Face Spaces.

**Datasets:** MVTec AD 2 (main benchmark) · MVTec AD (baseline, compare with published results) · VisA (second benchmark, supported by Anomalib) · held-out defect types (own split to test discovery and few-shot). Licence note in the spec: MVTec datasets are non-commercial — fine for a portfolio.

**Verified facts relevant to planning:** MVTec AD 2 is about 30.4 GB in total, with per-category downloads available, and its test data comes as test\_public (with ground truth), test\_private and test\_private\_mixed ([MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad-2/downloads)). Ground-truth evaluation of the private sets is only possible through MVTec's benchmark server ([IJCV paper](https://link.springer.com/article/10.1007/s11263-026-02743-0)), and its training and validation sets contain only anomaly-free images. Anomalib ships an MVTecAD2 datamodule with public / private / private\_mixed test types ([Anomalib docs](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/data/datamodules/image/mvtecad2.html)). **\[interpretation\]** We therefore report AUROC metrics on test\_public locally and treat leaderboard submission as optional; the good-only validation split is a natural conformal calibration set.

**Metrics (p.5–6):** image + pixel AUROC on MVTec AD 2, MVTec AD, VisA · conformal: target vs actual false-alarm rate, miss rate at that threshold · discovery: detection delay and cluster purity · few-shot accuracy at 5 / 10 shots + old-type accuracy after learning (forgetting) · latency and accuracy FP32 vs INT8, CPU vs GPU · expected cost per 10,000 parts vs fixed threshold (stated assumptions) + share sent to human review · counterfactual quality (defect removed, rest unchanged).

### 1.4 Integration requirements

1. New defect cluster in P2 → case automatically opened in P1.
2. Agent asks smart questions, returns a guaranteed cause set.
3. Engineer confirms (human-in-the-loop).
4. FMEA updated; knowledge graph and probabilities updated; vision side learns the new type (few-shot).

**\[interpretation\]** MVTec images carry no process data. To link them, the simulator assigns each inspected image a synthetic part ID, timestamp, station and material batch, and maps defect types to injected fault types (e.g. a new scratch cluster ← fixture misalignment). This linkage is synthetic and will be documented as such.

### 1.5 Roadmap (p.7) and deliverables (p.7–8)

1 P2 L1 → 2 P2 L2–3 → 3 P1 L1–2 → 4 P1 L3 → 5 simulator + scientist agent → 6 cost-aware decisions → 7 connect projects → 8 remaining advanced parts one at a time → 9 release + report + video. Tip: finish at most one or two signature moves per project first; strongest are scientist agent + simulator and cost-aware decisions.

**Final deliverables:** working dockerised system · simulator and 8D evaluation set on Hugging Face · 4–6 page technical report (problem, method, experiments, results, what failed) · “What I tried and why it failed” README section · 2-minute demo video at the top of README · architecture diagram + results table in README · live demo + GitHub links · LinkedIn post on one finding · CV bullets with real numbers only.

## 2. Technical architecture (Task 2)

### 2.1 System architecture

&#91;embedded content: system architecture · 2 services, MCP server, 4 stores, simulator\]

The dashboard talks only to the two service APIs, never to databases directly. The vision service publishes a `new_defect_cluster` event; the diagnosis service consumes it and opens a case. The agent reaches the mock quality/MES system only through MCP tools, which is how it would reach a real QMS in a factory. All services share one data layer; the simulator writes Parquet files that DuckDB queries in place.

**Architectural style:** a modular monolith per service (clean layers inside each service), two deployable services, shared contracts in a `common` package. No Kubernetes, Kafka, Celery or Redis — none is needed at this scale, and each would add operational weight without a measurable benefit.

### 2.2 Component architecture

**Project 2 — vision service components**

| Component | Responsibility | Main algorithms / libraries |
| --- | --- | --- |
| `data` | Dataset adapters for MVTec AD, MVTec AD 2, VisA; held-out defect-type splits; manifest files with hashes | Anomalib datamodules, own split logic |
| `detectors` | Unsupervised anomaly models behind one interface (`fit`, `score`, `anomaly_map`) | PatchCore (memory bank, coreset), EfficientAD (student–teacher) |
| `supervised` | Supervised baseline classifier for comparison | Fine-tuned ResNet-18/50 or ViT-S via timm |
| `calibration` | Split-conformal threshold per category and model version | Own implementation (quantile with finite-sample correction) |
| `decision` | Accept / reject / human-review under stated costs | Probability calibration (Platt / isotonic) + expected-cost minimisation (NumPy) |
| `explain` | Anomaly maps, Grad-CAM for the supervised model, VLM text description, counterfactual “repaired” image | Anomalib maps, pytorch-grad-cam, VLM via API or local, diffusers inpainting |
| `embeddings` | Feature extraction for anomalous regions | DINOv2 ViT-S/14 (384-d) |
| `discovery` | Unknown-anomaly buffer, clustering, new-cluster alerts, labelling queue | UMAP + HDBSCAN |
| `fewshot` | New classes from 5–10 shots without forgetting | Prototypical classifier + prototype memory + replay buffer |
| `edge` | Export, INT8 quantisation, latency/accuracy benchmark | ONNX, ONNX Runtime, OpenVINO / NNCF |
| `monitoring` | Score and embedding drift reports | Evidently |
| `api` | Inspection, calibration, cluster, few-shot, model endpoints | FastAPI |

**Project 1 — diagnosis service components**

| Component | Responsibility | Main algorithms / libraries |
| --- | --- | --- |
| `ingestion` | Load NHTSA recalls/complaints, AI4I; generate 8D + FMEA (DE/EN) with provenance flags | pandas, LLM gateway |
| `taxonomy` | Fixed cause / failure-mode taxonomy shared by RAG, BN, conformal and simulator | Versioned YAML |
| `retrieval` | Hybrid search over past cases | BM25 + dense embeddings (pgvector) + reciprocal-rank fusion + cross-encoder reranker |
| `graph` | Neo4j repository: write entities, expand neighbourhoods, update after confirmation | Neo4j Python driver, Cypher |
| `novelty` | Decide “no close past case” | Max-similarity threshold calibrated on held-out new problems |
| `scoring` | Probability vector over taxonomy causes from all evidence | Bayesian-network posterior + retrieval votes + test outcomes, then calibration |
| `conformal` | Cause sets with coverage guarantee | Split conformal, LAC and APS scores (own implementation, MAPIE as cross-check) |
| `bayes` | Cause–symptom network; expected-information-gain question selection | pgmpy (variable elimination), NumPy/SciPy entropy |
| `scientist` | Hypothesis → parameterised SQL on DuckDB → statistical test → verdict + evidence | SciPy, statsmodels, Benjamini–Hochberg correction |
| `reporting` | 8D report (D0–D8) drafting and completeness check | LLM + Pydantic schema validation |
| `fmea` | Compare confirmed cause with FMEA; propose new entries; approval workflow | Graph lookups + LLM drafting + human approval |
| `feedback` | Update graph and BN counts after confirmation; add the case to the candidate pool for the next versioned calibration set | Dirichlet count updates, graph upserts |
| `llm` | One gateway for API and local models: retries, caching, budget limit, tracing | Provider SDKs, Ollama HTTP, Langfuse |
| `agent` | Orchestration as a LangGraph state machine with human-in-the-loop interrupts | LangGraph + Postgres checkpointer |
| `mcp_server` | Mock QMS/MES exposed as MCP tools | Official MCP Python SDK |
| `evaluation` | Labelled test set, metrics, LLM-as-judge validation | Ragas or DeepEval, own metrics, Cohen’s kappa |

**Factory simulator components:** line model (stations, machines, buffers), material batches and suppliers, sensor generators (temperature, spindle load, vibration, tool wear), shifts/operators, fault injector (scenario YAML: fault type, onset, magnitude, affected station), ground-truth ledger (stored separately so the agent cannot read it), exporters (Parquet), image linker (assigns MVTec images to simulated parts for the integration demo).

**Shared (`common`):** Pydantic schemas for events and API contracts, settings, logging, database clients, ID generation.

### 2.3 Backend architecture

Each service follows the same four layers:

1. **API layer** (FastAPI routers): request validation, auth, response models. No business logic.
2. **Application layer** (use cases such as `InspectImage`, `RunDiagnosis`, `ConfirmCause`): orchestrates domain objects and adapters; owns transactions.
3. **Domain layer:** pure Python — conformal calibrators, decision rules, information gain, hypothesis-test selection. No I/O, so it is unit-testable in milliseconds.
4. **Infrastructure adapters** behind interfaces (ports): `VectorStore`, `GraphStore`, `LLMClient`, `AnomalyModel`, `ProcessDataStore`. Swapping pgvector for Qdrant, or the API model for Ollama, touches one adapter.

**Long-running work** (agent runs, clustering, inpainting, benchmarks) goes to a `jobs` table processed by a worker process in the same image. The API returns `202 Accepted` with a job ID; the dashboard polls. This avoids Celery/Redis while keeping requests fast.

**Agent graph (LangGraph):** `intake` → `extract_symptoms` (LLM → structured symptoms mapped to taxonomy) → `retrieve` (hybrid RAG) → `kg_expand` → `novelty_check` → `score_causes` → loop {`select_question` (max EIG) → **interrupt** for the engineer’s answer → `update_posterior`} until entropy or question budget is reached → `test_hypotheses` (scientist) → `conformal_set` → `draft_8d` → `fmea_compare` → `open_ticket` (MCP) → **interrupt** for confirmation → `feedback_update`. State is checkpointed in Postgres so a case can pause for hours and resume.

**A rule that matters for correctness:** the LLM extracts, explains and drafts; it never produces the final probabilities. Probabilities come from the Bayesian network, retrieval and data tests, then pass through calibration and conformal prediction. LLM confidence statements are not calibrated, so they cannot carry a statistical guarantee.

**Security and robustness:** API-key header auth on every non-health endpoint (keys from environment); CORS restricted to the dashboard origin; upload limits and image type checks; read-only DuckDB connections for the scientist agent with parameterised SQL templates (no free-form LLM SQL in the base version); secrets only in `.env` (never committed); `/health` and `/ready` endpoints; structured JSON logs with a request/case ID; versioned routes (`/v1`).

### 2.4 Database design

Four stores, each chosen for one job. Postgres is the system of record; Neo4j holds relationships the agent traverses; DuckDB queries simulator Parquet files without loading them into a server; MLflow tracks experiments.

**PostgreSQL 16 + pgvector — schemas and main tables**

| Schema | Table | Key columns (abridged) |
| --- | --- | --- |
| `diagnosis` | `cases` | id, source (nhtsa / complaint / vision / simulator / manual), title, description, symptoms (jsonb), status, novelty\_flag, confirmed\_cause\_id, created\_at |
|  | `case_chunks` | id, case\_id, text, language, embedding `vector(d)`, tsv (`tsvector` for BM25-style search), provenance (real / generated) |
|  | `cause_taxonomy` | id, code, name\_en, name\_de, category, parent\_id, test\_template |
|  | `case_causes` | case\_id, cause\_id, score, rank, in\_conformal\_set, calibration\_run\_id |
|  | `questions_asked` | case\_id, question\_id, answer, eig, entropy\_before, entropy\_after |
|  | `hypothesis_tests` | id, case\_id, cause\_id, sql\_template, params, test\_name, statistic, p\_value, p\_adjusted, effect\_size, verdict, evidence (jsonb) |
|  | `reports_8d` | case\_id, version, language, content (jsonb D0–D8), completeness |
|  | `fmea_entries` | id, item, function, failure\_mode, effect, cause, severity, occurrence, detection, action\_priority, source |
|  | `fmea_proposals` | id, case\_id, proposal (jsonb), status (pending / accepted / rejected), reviewer, decided\_at |
|  | `calibration_runs` | id, method, alpha, qhat, n\_calibration, dataset, created\_at |
| `vision` | `inspections` | id, image\_uri, category, model\_version, score, threshold, p\_defect, decision (accept / reject / review), defect\_class, cluster\_id, part\_id, created\_at |
|  | `anomaly_embeddings` | inspection\_id, embedding `vector(384)` |
|  | `clusters` | id, status (candidate / alerted / labelled), size, first\_seen, last\_seen, stability |
|  | `defect_classes` | id, name, n\_shots, prototype `vector(384)`, created\_from\_cluster |
|  | `calibrations` | model\_version, category, alpha, qhat, n\_calibration |
|  | `cost_configs` | id, cost\_false\_reject, cost\_escape, cost\_review, currency, note (“assumption”) |
| `qms` (MCP-only) | `tickets`, `ticket_events` | id, case\_id, title, severity, status, assignee, history |
|  | `batches`, `machines` | master data mirrored from the simulator |
| `platform` | `events` (outbox) | id, type, payload (jsonb), status, attempts, created\_at, processed\_at |
|  | `jobs` | id, kind, status, input, result, error |
|  | LangGraph checkpoint tables | managed by `langgraph-checkpoint-postgres` |

FMEA columns follow the AIAG–VDA structure (severity, occurrence, detection and action priority); a classic RPN can be computed for comparison. The embedding dimension `d` for text is fixed once the embedding model is chosen in step 3.

**Neo4j knowledge graph model**

- Nodes: `Part`, `Component`, `Process`, `Machine`, `FailureMode`, `Cause`, `Effect`, `Symptom`, `FMEAEntry`, `Case`, `DefectType`, `Batch`.
- Relationships: `(Part)-[:HAS_COMPONENT]->(Component)`, `(Component)-[:PRODUCED_BY]->(Process)`, `(Process)-[:RUNS_ON]->(Machine)`, `(Component)-[:CAN_FAIL_AS]->(FailureMode)`, `(FailureMode)-[:CAUSED_BY]->(Cause)`, `(FailureMode)-[:LEADS_TO]->(Effect)`, `(FailureMode)-[:SHOWS]->(Symptom)`, `(FMEAEntry)-[:DOCUMENTS]->(FailureMode)`, `(Case)-[:EXHIBITS]->(Symptom)`, `(Case)-[:CONFIRMED_CAUSE]->(Cause)`, `(Case)-[:SIMILAR_TO {score}]->(Case)`, `(DefectType)-[:INDICATES]->(FailureMode)`.
- Every node and edge carries provenance: `source` (nhtsa / ai4i / generated / simulator / engineer), `confidence`, `created_at`. A failure mode with a `CAUSED_BY` edge but no `DOCUMENTS` edge from an FMEA entry is exactly what the Living FMEA layer detects.

**DuckDB over Parquet (simulator output)**

`parts` (part\_id, station path, batch\_id, timestamps, final\_quality) · `sensor_readings` (machine\_id, ts, temperature, spindle\_load, vibration, tool\_wear) · `station_events` · `batches` (supplier, hardness) · `shifts` · `ground_truth` (injected fault, onset, magnitude) — **stored in a separate directory that the agent’s read-only connection cannot see**, so evaluation cannot leak.

**MLflow:** local file store or SQLite backend, artifact directory on disk; one experiment per pipeline (detector, conformal, discovery, fewshot, edge, diagnosis-eval).

### 2.5 ML pipeline architecture

**P2-A Detector training and evaluation:** dataset + category → preprocessing (resize, normalise) → fit detector (PatchCore builds a coreset memory bank from good images; EfficientAD trains a student–teacher pair) → score the good-only validation split → conformal threshold → evaluate on test (image AUROC, pixel AUROC, realised false-alarm rate, miss rate) → log to MLflow → register model version.

**P2-B Conformal threshold:** split-conformal threshold per category on good-only calibration images that were **not** used to fit the detector (PatchCore’s memory bank, EfficientAD’s training set). Full mathematics, assumptions and limits are in §2.8.

**P2-C Cost-aware decisions:** calibrate scores to a defect probability p (Platt or isotonic, on a labelled split kept apart from both the conformal calibration set and the test set). Expected cost of each action: accept = p·C\_escape, reject = (1−p)·C\_false\_reject, review = C\_review (+ residual human error). **Revision 2 constraint:** in the guaranteed mode, a part may only be *flagged* (rejected or sent to review) if its score exceeds the conformal threshold q̂; the cost rule chooses between reject and review inside that region, and below q̂ the part is accepted. This keeps “at most α of good parts flagged” true by construction. An unconstrained cost-optimal policy is also evaluated, but reported as having **no** false-alarm guarantee. Results: expected cost per 10,000 parts (guaranteed policy, unconstrained policy, fixed threshold), review share, and sensitivity over cost ratios. Euro values are stated assumptions, never presented as measured.

**P2-D Discovery:** anomalous inspection → mask from anomaly map → crop → DINOv2 embedding → distance to known defect prototypes; above a threshold the sample goes to the unknown buffer → periodic UMAP (fixed random seed) + HDBSCAN → a cluster that reaches the minimum size and stays stable across runs raises an alert and enters the labelling queue. Evaluation streams test images in time order with one held-out defect type injected from a known point, measuring detection delay (samples until alert) and purity / adjusted Rand index.

**P2-E Few-shot + continual:** prototype = mean of L2-normalised embeddings of k labelled shots; classify by cosine similarity to prototypes. Old prototypes are kept in memory (no forgetting by construction); if a trainable head is added, a replay buffer of old-class embeddings is mixed into each update. Evaluated over repeated random draws of 5 and 10 shots (mean ± std), plus old-class accuracy before vs after.

**P2-F Edge:** export (Anomalib export or `torch.onnx`) → static INT8 post-training quantisation with calibration images (ONNX Runtime `quantize_static`; OpenVINO via NNCF) → benchmark p50/p95 latency, throughput, model size, AUROC change.

**P2-G Counterfactual (“how it should look”):** dilated defect mask → diffusion inpainting → checks: anomaly score of the repaired image falls below the threshold; similarity outside the mask stays high (SSIM / LPIPS). A classical inpainting baseline (OpenCV) is compared, so the diffusion model has to earn its cost.

**P1-A Knowledge ingestion:** NHTSA recalls → clean, filter to a manageable set of component groups → map to taxonomy → LLM generates 8D + FMEA rows (cached, `provenance=generated`, a sample reviewed by hand) → chunk + embed → pgvector; entities → Neo4j.

**P1-B Bayesian network:** structure defined from the taxonomy (causes → failure modes → observable symptoms); conditional probabilities estimated from AI4I 2020 (its failure types and discretised air/process temperature, speed, torque, tool wear) and simulator data with a Bayesian estimator (Dirichlet prior). Question selection: for each unasked question Q, EIG(Q) = H(C | e) − Σ\_q P(q | e) · H(C | e, q); ask the maximum.

**P1-C Scoring + conformal:** evidence → probability vector over taxonomy causes → nonconformity score (LAC or APS) → threshold from calibration cases → cause set. The scorer must be fitted on data disjoint from the calibration cases; mathematics, assumptions and the reasons held-out new problems carry no guarantee are in §2.8.

**P1-D Scientist agent:** each taxonomy cause has a test template. Examples: tool wear → defect rate above vs below a wear level (Fisher exact / chi-square); material batch → defect rate by batch (chi-square); temperature drift → temperature of defective vs good parts in the window (Mann–Whitney U); onset in time → before/after comparison. p-values are corrected across hypotheses (Benjamini–Hochberg); verdict = supported / rejected / inconclusive (too little data), always with the query, the statistic and a plot.

**P1-E Evaluation:** labelled cases split into dev / calibration / test / held-out new problems → run agent → metrics. CI runs a small fixed subset with recorded LLM responses, so it is deterministic and costs nothing; the full evaluation is run manually with a budget limit.

**P1-F Own small model:** training pairs (case → structured diagnosis + 8D) from generated and simulator cases → QLoRA on a 1.5–3B instruct model on a free cloud GPU → serve via Ollama (GGUF) → same evaluation as the API model; cost from token prices vs local compute time.

### 2.6 API communication architecture

| Channel | From → to | Protocol | Notes |
| --- | --- | --- | --- |
| User actions | Dashboard → services | REST/JSON over HTTP, API key | Long jobs return 202 + job ID |
| New-defect event | Vision → diagnosis | Transactional outbox in Postgres, polled by the diagnosis worker | At-least-once delivery; handler idempotent on event ID |
| Tool calls | Agent → MCP server | MCP (stdio in development, streamable HTTP in Docker) | Tools: `create_ticket`, `update_ticket`, `get_ticket`, `get_batch_info`, `query_process_data`, `search_fmea` |
| LLM calls | Diagnosis → API model / Ollama | HTTPS / local HTTP | Through the `llm` gateway only (budget, cache, trace) |
| Data | Services → stores | SQL, Bolt (Neo4j), DuckDB in-process | DuckDB opened read-only for the agent |

**Main endpoints (v1)**

- Vision: `POST /v1/inspections` (image → score, map, decision) · `GET /v1/inspections/{id}` · `POST /v1/calibrations` · `GET /v1/clusters` · `POST /v1/clusters/{id}/labels` · `POST /v1/defect-classes` (few-shot) · `GET /v1/models` · `GET /v1/benchmarks/edge` · `POST /v1/inspections/{id}/counterfactual`
- Diagnosis: `POST /v1/cases` · `GET /v1/cases/{id}` · `POST /v1/cases/{id}/run` · `GET /v1/cases/{id}/next-question` · `POST /v1/cases/{id}/answers` · `GET /v1/cases/{id}/cause-set` · `GET /v1/cases/{id}/hypotheses` · `GET /v1/cases/{id}/report-8d?lang=de|en` · `POST /v1/cases/{id}/confirm` · `GET /v1/fmea/proposals` · `POST /v1/fmea/proposals/{id}/decision` · `GET /v1/graph/subgraph?case_id=`
- Both: `GET /health`, `GET /ready`, `GET /v1/jobs/{id}`, OpenAPI docs.

### 2.7 Data flows

&#91;embedded content: integrated quality loop · 7 steps, 1 human gate\]

**Inspection flow:** image upload → validate → detector score + anomaly map → compare with conformal threshold → cost decision (accept / reject / review) → if anomalous: embedding → known class or unknown buffer → store inspection → return result with heatmap.

**Diagnosis flow:** case text (or vision event) → structured symptoms → hybrid retrieval + graph expansion → novelty check (if novel, say so and rely on BN + data tests) → cause scores → question loop → hypothesis tests on DuckDB → conformal cause set → 8D draft + FMEA comparison → ticket via MCP → wait for confirmation.

**Confirmation flow:** confirmed cause → graph edge `CONFIRMED_CAUSE` → BN count update → case added to the retrieval index and to a candidate pool for the next versioned calibration set (never directly to the active one, §2.8) → FMEA proposal if the failure mode is undocumented → for vision-originated cases the cluster becomes a named defect class (few-shot prototype).

**Simulator flow:** scenario YAML (seed, line config, fault list) → SimPy run → Parquet tables + separate ground-truth ledger → cases generated from observed symptoms → used for evaluation and for the scientist agent’s tests.

### 2.8 Conformal prediction — mathematics, assumptions and limits

Both projects use **split conformal prediction**. This subsection is the reference for the implementation, the tests and the claims we are allowed to make.

**Setup.** A score function s is fixed in advance (trained on separate data). We compute calibration scores S₁, …, Sₙ on n calibration examples and Sₙ₊₁ on a new example.

**Assumption A1 — exchangeability.** The joint distribution of (S₁, …, Sₙ₊₁) is unchanged by any permutation. It holds when calibration and test examples are i.i.d. draws from the same distribution and s does not depend on them. Every guarantee below depends on A1; nothing else about the model needs to be correct.

**Finite-sample threshold.** Let k = ⌈(n+1)(1−α)⌉ and q̂ = S₍ₖ₎, the k-th smallest calibration score. If k > n, no finite threshold exists (q̂ = +∞), so we need n ≥ ⌈1/α⌉ − 1: at least **49** calibration images for α = 0.02 and **9** calibration cases for α = 0.10. The code raises an error instead of silently returning a threshold when n is too small.

**Marginal guarantee.** Under A1, P(Sₙ₊₁ ≤ q̂) ≥ 1 − α. If scores are almost surely distinct (no ties), also P(Sₙ₊₁ ≤ q̂) ≤ 1 − α + 1/(n+1). The probability is taken over *both* the random calibration set and the new example. That is what “marginal” means.

**What marginal does not give.**

- *Calibration-conditional coverage:* for one fixed calibration set the realised coverage is itself random. With continuous scores it follows Beta(k, n+1−k). With small n a particular deployed threshold can be noticeably worse than 1−α. We therefore also implement a conservative **(α, δ) threshold** that chooses k so that P(realised coverage ≥ 1−α) ≥ 1−δ under that Beta distribution, and report both.
- *Conditional coverage:* there is no guarantee for a particular product variant, defect type or case type. We calibrate separately per vision category and per data source (Mondrian / group-wise conformal), which guarantees coverage within each group but needs enough data per group.
- *Ties:* discrete scores make coverage conservative (≥ 1−α still holds). The upper bound no longer holds.

**Project 2 application (false-alarm guarantee).** Score = anomaly score; calibration set = good images only, disjoint from the images used to fit the detector. *Why disjoint matters:* PatchCore’s memory bank contains its training images, so their scores are near zero; calibrating on them would set q̂ too low and the realised false-alarm rate would exceed α. Statement we can make: **P(a new good part scores above q̂) ≤ α**, per category, under A1. There is **no** guarantee on the miss rate (defects below q̂); it is measured and reported. With the constrained decision policy (§2.5 P2-C) the same bound holds for “good part flagged”.

**Project 1 application (cause sets).** For case x and cause y in the taxonomy, LAC score s(x, y) = 1 − p̂(y | x); set C(x) = {y : s(x, y) ≤ q̂}, with q̂ computed from the true causes of the calibration cases. Guarantee: P(true cause ∈ C(X)) ≥ 1 − α, marginally. LAC gives the smallest average sets but may return an empty set (shown to the engineer as “no cause reaches the threshold → treat as novel / escalate”). APS sorts causes by probability and adds them until the cumulative mass passes the threshold, so sets adapt to case difficulty. We use the non-randomised version (slightly conservative, easier to explain).

**Project 1 assumptions that need explicit engineering:**

1. *Disjointness:* BN parameters, retrieval index contents used for scoring, fusion weights and probability calibration are fitted without the calibration cases. When scoring a calibration case, retrieval excludes that case itself.
2. *Closed label space:* the guarantee covers cases whose true cause is in the taxonomy. Cases with out-of-taxonomy causes are counted separately as a novelty-detection problem.
3. *Held-out “new problems” are not exchangeable with the calibration set by construction.* Coverage on them is reported as a stress test, **never as a guaranteed number**. *(Revision 2 correction to step 4.)*
4. *Real vs simulated cases* come from different distributions and are calibrated and reported separately.
5. *Feedback loop:* engineer-confirmed cases are not a random sample, and confirmation can be influenced by the agent’s own suggestion. They are **not** appended to the calibration set automatically. Calibration sets are versioned and frozen; after the scorer is updated (e.g. BN counts), recalibration uses fresh cases that were not used for that update.
6. *Small n:* with \~30–40 real calibration cases, realised coverage varies widely. We report Clopper–Pearson intervals and the Beta(k, n+1−k) spread, and make headline coverage claims only on the large simulator set.

**Limits under distribution shift.** If the test distribution changes (new lighting, new supplier, tool change), A1 fails and the guarantee can silently break. Options, in order of cost:

- *Monitor + recalibrate (Tier A):* track the realised false-alarm rate on parts that humans reviewed and the score distribution (Evidently when added). Trigger recalibration when the Clopper–Pearson interval of the realised rate excludes α.
- *Weighted conformal (Tier C):* restores validity under covariate shift if the likelihood ratio between test and calibration inputs is known or can be estimated well. It is not planned for Tier A.
- *Adaptive conformal inference (Tier C):* updates αₜ online (αₜ₊₁ = αₜ + γ(α − errₜ)) and guarantees the *long-run average* error rate, not a per-part probability. It needs label feedback (from human review).

**Planned shift experiment (step 2):** apply the in-distribution threshold to shifted test data. We use MVTec AD 2 lighting variants where the condition can be identified in test\_public. Otherwise we apply controlled synthetic photometric shifts (brightness, contrast, colour temperature). We report how far the realised false-alarm rate moves from α. This is a planned measurement, not a predicted result.

**How the implementation is tested:** unit tests for the index rule (k > n ⇒ error; n = 49, α = 0.02 ⇒ k = 49); property tests on simulated exchangeable scores (over 10,000 trials, mean coverage lies in \[1−α, 1−α+1/(n+1)\] within Monte-Carlo error, and the realised-coverage distribution matches Beta(k, n+1−k) by a KS test); a leakage test asserting calibration IDs ∩ fit IDs = ∅.

### 2.9 Feature modularity — running without advanced features

The integrated loop (Tier A) must run with a **core install only**. Advanced features are optional extras that can be absent without breaking anything.

- **Optional dependency extras.** Vision: `discovery` (umap-learn), `edge` (onnx, onnxruntime), `openvino` (openvino, nncf), `monitoring` (evidently), `explain` (pytorch-grad-cam), `vlm-local` (transformers, accelerate), `experiments` (mlflow). Diagnosis: `anthropic` / `openai` (provider SDKs), `eval` (ragas or deepeval), `tracing` (langfuse), `experiments` (mlflow). Diffusion inpainting lives only in the GPU notebooks; the service serves precomputed results plus an OpenCV baseline (OpenCV is already core).
- **Core needs no extras for the minimal loop.** HDBSCAN has been part of scikit-learn since version 1.3 ([scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)), and DINOv2 backbones are loaded through timm, which is already core. So the step-7 discovery trigger needs neither umap-learn nor transformers.
- **Capability registry.** At startup each feature checks that (a) it is enabled in settings, (b) its extra imports successfully, and (c) its required artifact exists (e.g. an INT8 ONNX file). `GET /v1/capabilities` reports each feature as `enabled`, `disabled`, `not_installed` or `missing_artifact`. Endpoints of unavailable features return **501** with the reason. The dashboard reads capabilities and shows those pages as “not installed” with the install command, so no button ever produces fake output.
- **Import rule.** Tier A code never imports Tier B/C modules directly; it calls them through registry interfaces. A CI lint rule blocks direct imports.
- **Containers.** Docker images take an `EXTRAS` build argument (default: core). Compose profiles: `core` (Postgres, Neo4j, vision, diagnosis, MCP server, dashboard), `llm-local` (Ollama), `mlflow-ui`, `tracing`. The minimal integrated system is `core` + one LLM backend (local Ollama or an approved API key).
- **Proof in CI.** A *core-only* job installs each service without extras and runs its tests plus the step-7 end-to-end test. A *full* job installs all extras. Both must pass.

### 2.10 Dependency isolation and locking (Revision 2 correction)

**Problem found in review.** Revision 1 proposed one uv workspace. A workspace shares one lockfile so all members operate with a consistent set of dependencies ([uv docs](https://docs.astral.sh/uv/concepts/workspaces)). The uv documentation says workspaces are not suited when members have conflicting requirements or need separate virtual environments, and recommends path dependencies in that case ([uv docs mirror](https://git.joshthomas.dev/mirrors/uv/raw/branch/main/docs/concepts/projects/workspaces.md)). Workspaces also enforce a single requires-python, the intersection of all members’ values ([uv docs](https://docs.astral.sh/uv/concepts/projects/workspaces)). That contradicts our requirement that vision and diagnosis install and build independently.

**Decision.** Each component is an **independent uv project** with its own `pyproject.toml`, `uv.lock` and `.venv`: `common`, `vision`, `diagnosis`, `mcp_server`, `simulator`, `dashboard`, and `tests/e2e`. Services depend on `common` through an editable path dependency (`iqs-common = { path = "../common", editable = true }` under `[tool.uv.sources]`).

**Rules that keep this conflict-free:**

- `common` depends only on pydantic, pydantic-settings, SQLAlchemy, psycopg and structlog, with broad compatible ranges. It never depends on ML libraries, so it cannot cause conflicts in either service.
- No service imports another service’s package. Services communicate over HTTP, the outbox table and MCP. The simulator is imported by nobody; the agent reads its Parquet output.
- PyTorch comes from the CPU wheel index in both `vision` and `diagnosis` (sentence-transformers also pulls in torch), declared as an explicit index in each `pyproject.toml`. GPU wheels are used only in the cloud notebooks, which use the notebook runtime’s own torch.
- `uv lock --check` runs per project in CI, so a stale lockfile fails the build.
- Docker builds use the repository root as context, copy only `packages/common` and the one service, and run `uv sync --frozen --no-dev` (plus `--extra` from `EXTRAS`) in a multi-stage build. The final image contains only that service’s `.venv`.

**Trade-off accepted.** Shared libraries (e.g. pydantic) may resolve to different versions in different services. This is controlled by contract tests: event and API schemas in `common` are serialised to JSON and validated in both services’ test suites, and the e2e suite tests the real HTTP boundaries.

## 3. Development environment (Task 3)

### 3.1 Required software

Your machine runs Windows, so the recommended setup is **WSL2 (Ubuntu) + Docker Desktop with the WSL2 backend**. Keep the repository inside the WSL filesystem (e.g. `~/code/...`), not under `/mnt/c`, because file I/O across the boundary is slow and causes path and line-ending problems.

| Tool | Status | Purpose |
| --- | --- | --- |
| WSL2 + Ubuntu LTS | Essential | Linux environment matching Docker and CI |
| Docker Desktop (WSL2 backend) + docker compose | Essential | Postgres, Neo4j, services, reproducible runs |
| Git + GitHub account | Essential | Version control, Actions CI |
| `uv` | Essential | Python versions, per-project lockfiles and environments, fast installs |
| VS Code + WSL and Python extensions | Recommended | Editing and debugging inside WSL |
| Ollama | Essential from step 3 | Local LLM (on-prem option), also for cost-free development |
| NVIDIA driver + CUDA in WSL | Only if you have an NVIDIA GPU | GPU training locally |
| Google Colab / Kaggle notebooks | Essential if no local GPU | EfficientAD training, LoRA fine-tuning, diffusion inpainting |

### 3.2 Python version

**Python 3.11**, pinned in `.python-version`, in every project’s `requires-python` (`>=3.11,<3.12` initially) and in the Docker base images. Reason: it has the least friction across the heaviest dependencies (PyTorch, Anomalib, OpenVINO, pgmpy, LangGraph) at the same time. Because each project now has its own lockfile (§2.10), a later move to 3.12 can happen one service at a time. Exact package versions are not chosen in this blueprint: they are resolved and frozen in each project’s `uv.lock` in step 0 and recorded in TECHNICAL\_DECISIONS.md.

### 3.3 Dependencies by package

| Project (own lockfile) | Core (always installed) | Optional extras (installed only when that feature is used) |
| --- | --- | --- |
| `common` | pydantic, pydantic-settings, SQLAlchemy 2 + psycopg, pgvector (Python), structlog — broad version ranges, no ML libraries | — |
| `vision` | torch + torchvision (CPU index), timm (also provides DINOv2 backbones), anomalib, scikit-learn ≥ 1.3 (includes HDBSCAN), numpy, opencv-python-headless, fastapi, uvicorn, python-multipart | `discovery`: umap-learn · `edge`: onnx, onnxruntime · `openvino`: openvino, nncf · `monitoring`: evidently · `explain`: pytorch-grad-cam · `vlm-local`: transformers, accelerate · `experiments`: mlflow |
| `diagnosis` | langgraph, langgraph-checkpoint-postgres, httpx (also used for Ollama), neo4j driver, rank-bm25, sentence-transformers (embeddings + cross-encoder reranker; torch from CPU index), pgmpy, scipy, statsmodels, duckdb, pandas, pyarrow, mcp (client), fastapi, uvicorn | `anthropic` / `openai`: provider SDK · `eval`: ragas or deepeval · `tracing`: langfuse · `experiments`: mlflow |
| `mcp_server` | mcp (official Python SDK), SQLAlchemy via `common` | — |
| `simulator` | simpy, numpy, pandas, pyarrow, pyyaml | `release`: datasets, huggingface\_hub |
| `dashboard` | streamlit, httpx, plotly, streamlit-agraph or pyvis | — |
| `tests/e2e` | pytest, httpx | — |
| `finetune/` (cloud notebooks, not a service) | — | transformers, peft, trl, bitsandbytes, datasets, diffusers |
| dev group (each project) | pytest, pytest-cov, hypothesis, testcontainers, mypy | ruff runs through `uvx` with a pinned version, so it needs no project environment |

MLflow moved out of the service images into the `experiments` extra: training and evaluation scripts need it, the running services do not.

Streamlit is chosen as the single UI (the spec allows Gradio or Streamlit); one framework keeps the dashboard consistent, and Streamlit’s multipage apps fit the ten required views.

### 3.4 Hardware requirements

| Profile | CPU | RAM | Disk free | What runs |
| --- | --- | --- | --- | --- |
| Minimum | 4 cores | 16 GB | \~120 GB SSD | Everything except GPU-heavy training; run services selectively (compose profiles) |
| Recommended | 8+ cores | 32 GB | \~200 GB SSD | Full stack at once: Postgres, Neo4j, both services, Ollama 3B–7B, PatchCore |

RAM is the tightest resource: Neo4j, Postgres, an Ollama model and a PyTorch process together can exceed 16 GB. Compose profiles (`core`, `llm-local`, `mlflow-ui`, `tracing`; §2.9) let you start only what a phase needs, and Neo4j’s heap is capped in config.

### 3.5 GPU requirements

| Workload | CPU feasible? | Plan |
| --- | --- | --- |
| PatchCore fit + inference (256 px) | Yes — no gradient training, but feature extraction is slow on large categories | Local CPU; coreset subsampling; start with a few categories |
| EfficientAD training | Slow on CPU | Colab/Kaggle GPU; export weights, infer on CPU |
| Supervised ResNet/ViT baseline | Small models feasible, slow | ResNet-18 locally, larger on free GPU |
| DINOv2 ViT-S/14 embeddings | Yes | Local CPU |
| Bayesian network, conformal, statistics, simulator | Yes | Local CPU |
| Ollama 3B–7B inference | Yes, slower | Local CPU (3B for fast iteration) |
| LoRA / QLoRA fine-tune 1.5–3B | No (practically) | Free cloud GPU notebook |
| Diffusion inpainting | Very slow on CPU | Precompute on free GPU; serve cached results + classical baseline live |
| INT8 benchmarks | Yes (CPU is the target) | Local; GPU numbers only if a GPU is available (Tier C otherwise) |

### 3.6 Storage requirements

| Item | Size | Notes |
| --- | --- | --- |
| MVTec AD 2 | About 30.4 GB whole; per-category downloads available ([MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad-2/downloads)) | Start with 2–3 categories |
| MVTec AD | Roughly 5 GB (verify at download) | 15 categories |
| VisA | A few GB (verify at download) | Downloaded only via the approval script (§3.9), never by Anomalib automatically |
| NHTSA recalls + complaints flat files | Verify at download; complaints is the larger file | Load a filtered subset into Postgres |
| AI4I 2020 | Small (10,000 rows) | — |
| Model weights (WideResNet / DINOv2-S / embeddings / reranker) | A few GB in total | Hugging Face cache |
| Ollama models (3B and 7B) | Several GB | Pull only what you use |
| Diffusion inpainting model | Several GB | Only on the GPU notebook if possible |
| Docker images + volumes | 15–25 GB estimate | Two ML images are the large ones |
| Simulator output, MLflow artifacts | 1–10 GB, grows | Retention policy in config |

Sizes marked “verify” are estimates, not measured; the real numbers go into `data/README.md` after download.

### 3.7 External APIs and services

| Service | Free / paid | Essential? | Use |
| --- | --- | --- | --- |
| LLM API (Anthropic, OpenAI or Azure OpenAI — pick one) | **Paid** per token | Essential for the large-vs-small comparison; development can run on Ollama | Diagnosis, 8D drafting, data generation, LLM-as-judge |
| VLM for defect descriptions | Paid (API) or free (local Qwen-VL, heavy on CPU) | Essential (P2 Layer 1) | Text explanation of a defect |
| Ollama | Free, local | Essential | Local model, on-prem story |
| MVTec / VisA / NHTSA / UCI downloads | Free (MVTec registration, non-commercial) | Essential | Datasets |
| Google Colab / Kaggle | Free tiers with GPU quotas | Essential if no local GPU | Training, fine-tuning, inpainting |
| Hugging Face Hub | Free | Essential in step 9 | Dataset release, model cache; Spaces demo optional |
| GitHub + Actions | Free for public repos | Essential | Code, CI |
| MLflow (local) | Free | Essential | Experiment tracking (chosen over W&B: no account, works offline) |
| Langfuse | Free cloud tier or self-host | Optional (Tier C) | LLM traces, cost, latency |
| MVTec benchmark server | Free | Optional | Private-test leaderboard |

**Cost controls (before any paid call):** one LLM gateway with a hard budget limit read from `.env`; response caching on disk; temperature 0 for evaluation; CI uses recorded responses only; every paid batch job prints its estimated cost and waits for confirmation. No paid service is used without your approval.

**On Langfuse:** self-hosting v3 means running its own Postgres, ClickHouse, Redis and S3-compatible storage, and a single-VM guide quotes at least 4 cores, 16 GiB RAM and 100 GiB disk for it ([Railway guide](https://railway.com/deploy/langfuse-v3-1.md)). That would double our local footprint, so we use the free cloud tier (if you accept sending traces there) or leave tracing to our own Postgres cost/latency log, with Langfuse added later.

### 3.8 Windows + WSL2 + cloud-GPU workflow (Revision 2)

**Hybrid development loop.** Databases (Postgres, Neo4j) run in Docker. The Python services run natively inside WSL through `uv run` with auto-reload, which gives faster iteration and a working debugger. The full `docker compose --profile core` stack is used for integration tests, e2e tests and demos. One `make dev` target starts the databases and both APIs.

**Resource limits.** WSL2 runs as a single VM, and by default it can use 50% of Windows memory ([NVIDIA WSL2 notes](https://docs.nvidia.com/nim/wsl2/latest/troubleshooting.html)). Docker Desktop’s WSL2 backend lives inside that same VM, so its containers share the same memory limit ([Docker forum](https://forums.docker.com/t/memory-limit-problem/150107)). We set the limits explicitly in `%UserProfile%\.wslconfig`. Changes take effect after `wsl --shutdown` ([guide](https://zahidhasan.github.io/2025/02/04/how-to-configure-ram-in-wsl-linux.html)).

```ini
# %UserProfile%\.wslconfig — starting values, tuned once your hardware is known
[wsl2]
memory=10GB      # 16 GB machine: leave ~6 GB for Windows; 32 GB machine: ~20GB
processors=6     # leave at least 2 logical cores for Windows
swap=8GB
```

**Memory budget on a 16 GB machine (estimates, to be measured in step 0):** Postgres \~0.5–1 GB · Neo4j heap capped at 1 GB + 0.5 GB page cache · diagnosis service \~1–2 GB (embedding + reranker models) · vision service 2–4 GB while fitting PatchCore · Ollama 3B quantised \~2–3 GB. **Rule:** never fit PatchCore and run Ollama at the same time on a 16 GB machine. Compose profiles and Make targets enforce this.

**Filesystem.** Keep the repository, datasets (`~/iqs-data`) and model caches (`~/iqs-models`) on the WSL ext4 filesystem, not under `/mnt/c`. If the system drive is small, move the WSL distribution to a larger drive (`wsl --export` / `wsl --import`) before downloading datasets.

**Ollama.** Installed inside WSL (Linux build), reachable at `localhost:11434` from WSL and at `host.docker.internal:11434` from containers. Models are pulled only through an approved Make target (§3.9).

**Local NVIDIA GPU (if present).** Install only the Windows NVIDIA driver, not a Linux driver inside WSL; CUDA is then available inside WSL. GPU use stays optional: every service has a CPU code path.

**Cloud GPU workflow (Colab / Kaggle).** The cloud runs the *same code and config* as local, not notebook-only code.

1. The notebook clones the repo at a tagged commit and installs the relevant project without torch (the runtime’s CUDA torch is used).
2. Data comes from your private Drive / private Kaggle dataset. MVTec data stays private because of its licence.
3. It runs the same `scripts/train_*.py --config ...` command as locally.
4. It writes a **run bundle**: checkpoint / ONNX file, metrics JSON, the resolved config, git SHA, package versions and hardware info.
5. You download the bundle; `scripts/import_run_bundle.py` validates it and registers it in local MLflow.

A cloud result counts in EXPERIMENTS.md only if its bundle carries the commit SHA and config.

### 3.9 Approval gates (Revision 2)

**Policy:** nothing large, paid or quota-consuming happens implicitly. I ask you in chat first, and the code enforces the gate so it cannot be bypassed by accident.

| Category | Needs your approval when | Enforcement in code |
| --- | --- | --- |
| Datasets | Always (any dataset) | `scripts/download_dataset.py <name>` prints source, licence, size from `configs/assets.yaml` and target path, then requires `--approve`. Services never download data. Our dataset adapters raise `DatasetNotFound` (with the command to run) instead of calling Anomalib’s `prepare_data()`, which downloads the dataset if it is not available ([Anomalib docs](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/data/datamodules/image/mvtecad2.html)). |
| Model weights | Not yet in the approved list, or larger than 200 MB | `configs/assets.yaml` allowlist (model ID, revision, expected size, licence). `scripts/fetch_models.py --approve` fills `~/iqs-models`. At runtime `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HOME`/`TORCH_HOME` point at the local cache, so missing weights raise an error instead of downloading. |
| Ollama models | Always | `make pull-llm MODEL=... APPROVE=1`; compose never pulls models automatically. |
| Paid LLM / VLM APIs | Always; batch jobs need a cost approval each time | Defaults: `LLM_BACKEND=ollama`, `PAID_API_ENABLED=false`. The gateway refuses paid calls unless enabled **and** budget remains (`LLM_BUDGET_EUR`, spend ledger in Postgres). Batch jobs (8D generation, full evaluation, LLM-as-judge) print estimated tokens and cost and need `--approve-cost <amount>`. CI never has API keys. |
| Cloud GPU runs | Always (free, but uses your quota) | I propose the notebook, the expected runtime and the outputs; you start it. |
| New third-party services / accounts | Always | Listed in TECHNICAL\_DECISIONS.md before use. |
| Docker base images | First `compose up` only | I list image names and approximate sizes before the first pull. |

Every approval is logged in `docs/APPROVALS.md` (date, item, size or cost, purpose).

## 4. Repository structure (Task 4)

A monorepo of **independent uv projects** (Revision 2, §2.10): one repository, one CI, but each deployable component has its own `pyproject.toml`, `uv.lock` and virtual environment, so vision and diagnosis can be installed, tested and built without each other. Working name: `industrial-quality-ai` (Python import prefix `iqs_`).

```text
industrial-quality-ai/
├── README.md                  # demo video, architecture, results table, “what failed”
├── CHANGELOG.md
├── Makefile                   # setup/test/lint per project; dev, up, down, smoke
├── ruff.toml                  # shared lint config (ruff via uvx, pinned)
├── .python-version            # 3.11
├── .env.example               # every setting, no secrets; paid APIs off by default
├── .pre-commit-config.yaml    # ruff, secret scan, large-file guard, uv lock --check
├── .github/workflows/
│   ├── ci.yml                 # matrix per project: lock check, lint, types, tests; core-only + full extras
│   ├── eval.yml               # deterministic evaluation subset on every push
│   └── docker.yml             # build images (core), compose smoke + e2e
├── docs/
│   ├── ARCHITECTURE.md  IMPLEMENTATION_PLAN.md  TECHNICAL_DECISIONS.md
│   ├── EXPERIMENTS.md   TESTING.md  KNOWN_ISSUES.md  APPROVALS.md
│   └── diagrams/
├── packages/
│   ├── common/      pyproject.toml  uv.lock  src/iqs_common/  tests/
│   ├── vision/      pyproject.toml  uv.lock  src/iqs_vision/  tests/
│   │   # src: data/ detectors/ supervised/ calibration/ decision/ explain/ embeddings/
│   │   #      discovery/ fewshot/ edge/ monitoring/ capabilities.py api/ worker/
│   ├── diagnosis/   pyproject.toml  uv.lock  src/iqs_diagnosis/  tests/
│   │   # src: ingestion/ taxonomy/ retrieval/ graph/ novelty/ scoring/ conformal/ bayes/
│   │   #      scientist/ reporting/ fmea/ feedback/ llm/ agent/ evaluation/ capabilities.py api/ worker/
│   ├── mcp_server/  pyproject.toml  uv.lock  src/iqs_mcp/  tests/
│   ├── simulator/   pyproject.toml  uv.lock  src/iqs_sim/  tests/
│   └── dashboard/   pyproject.toml  uv.lock  src/iqs_dashboard/  tests/
├── configs/
│   ├── assets.yaml            # approved datasets + models: source, revision, size, licence
│   ├── vision/  diagnosis/  simulator/scenarios/
│   ├── taxonomy/causes.yaml
│   └── costs/example_costs.yaml
├── scripts/                   # download_dataset, fetch_models, train_*, evaluate_*, benchmark_*,
│                              # import_run_bundle, smoke.sh
├── eval_sets/  diagnosis/  vision/   # small, versioned labels and split manifests
├── tests/
│   ├── e2e/        pyproject.toml  uv.lock   # talks to the running stack over HTTP only
│   └── fixtures/   # tiny synthetic images, recorded LLM responses
├── deploy/
│   ├── docker/                # one multi-stage Dockerfile per service, EXTRAS build arg
│   └── compose/               # profiles: core, llm-local, mlflow-ui, tracing
├── finetune/                  # Colab/Kaggle notebooks, own requirements, run-bundle export
├── notebooks/                 # exploration only
├── reports/  release/
└── (outside the repo, in WSL home)  ~/iqs-data  ~/iqs-models  ~/iqs-mlruns
```

| Directory | Purpose and rule |
| --- | --- |
| `packages/common` | Contracts shared by every service (event and API schemas, settings, DB sessions). No ML dependencies, broad version ranges. Changes need contract tests in both services. |
| `packages/vision` | All of Project 2. Domain logic (conformal, decisions, prototypes) has no I/O and is unit-tested. Advanced features live behind the capability registry and optional extras. |
| `packages/diagnosis` | All of Project 1, including the LangGraph agent and the LLM gateway with budget enforcement. |
| `packages/mcp_server` | Separate process exposing the mock QMS/MES as MCP tools; the agent changes “factory systems” only through it. |
| `packages/simulator` | Standalone library + CLI, releasable on its own, deterministic per seed. Imported by no service. |
| `packages/dashboard` | UI only. Talks to the APIs and reads `/v1/capabilities` to show unavailable features honestly. |
| `tests/e2e` | Own small project; tests the integrated system across real HTTP boundaries, so it never imports service code. |
| `configs/` | Every tunable value, plus `assets.yaml`, the single source for what may be downloaded. |
| `scripts/` | Thin CLIs over package code; download and paid scripts enforce approval flags. |
| `eval_sets/` | Your labels and split manifests with checksums. Large data never goes into git. |
| `deploy/` | Multi-stage images built from one service’s lockfile; compose profiles to save RAM. |
| `finetune/` | GPU-only notebooks that produce run bundles; not part of any service environment. |
| `docs/` | The seven required documents plus KNOWN\_ISSUES and APPROVALS. |
| `~/iqs-*` | Datasets, model caches and MLflow runs live outside the repo on the WSL filesystem. |

## 5. Implementation roadmap (Task 5)

The nine steps follow page 7 of the spec. A **Step 0 (foundation)** comes first, because no later step can be tested or reproduced without the repo, CI and Docker skeleton. Every step ends with a demonstrable result, a git tag and an updated IMPLEMENTATION\_PLAN.md.

**Two sequencing issues in the spec, and how they are resolved:**

- Step 7 (connect the projects) needs “camera finds a new defect cluster”, but full discovery is in step 8. Resolution: step 7 builds a minimal discovery trigger: an unknown-anomaly buffer plus scikit-learn’s HDBSCAN on DINOv2 (timm) embeddings. It is core-only, with no UMAP, no labelling UI and no evaluation claims. Step 8c turns it into the evaluated discovery layer.
- Step 7’s loop also mentions smart questions and FMEA updates (step 8). Resolution: step 7 runs the loop with the base agent (ranked causes, conformal set, 8D, confirmation); 8a and 8b then upgrade those stages inside the already-working loop.

**How completion is judged (Revision 2).** Every step has a **gate**: binary checks that pass or fail. It also has **recorded metrics**: numbers that must be measured, logged with a run ID and reported. Metrics are *not* performance targets, because results cannot be promised before they are measured. The exceptions are reproduction sanity checks and statistical validity checks, which have explicit tolerances. Every gate also includes the **universal gate**: CI green on `main` for all affected projects in both core-only and full-extras jobs, docs updated (ARCHITECTURE, TESTING, EXPERIMENTS, CHANGELOG, KNOWN\_ISSUES), checklist updated, git tag created.

### Step 0 — Foundation

- **Objectives:** a repo where setup, tests and the core stack work on your Windows + WSL2 machine and in CI.
- **Features:** six independent uv projects + `tests/e2e` (importable skeletons), settings via `.env` (paid APIs off), structured logging, Postgres + pgvector and Neo4j in compose, `/health`, `/ready` and `/v1/capabilities` on both services, `assets.yaml` + approval-enforcing download and model-fetch scripts (no downloads performed yet), CI matrix, pre-commit, the docs set, cause taxonomy v0 for review, `.wslconfig` recommendation tuned to your hardware.
- **Files:** `packages/*/pyproject.toml`, `packages/*/uv.lock`, `Makefile`, `.env.example`, `deploy/compose/docker-compose.yml`, `deploy/docker/*.Dockerfile`, `.github/workflows/{ci,docker}.yml`, `packages/common/src/iqs_common/{settings,logging,db,events}.py`, `packages/*/src/*/capabilities.py`, `scripts/{download_dataset,fetch_models,smoke.sh}`, `configs/{assets,taxonomy/causes}.yaml`, `docs/*.md`.
- **Dependencies:** WSL2, Docker Desktop, uv, Git, GitHub repository.
- **Testing:** settings and capability-registry unit tests; API tests for health/capabilities; tests that the download/fetch scripts refuse to run without `--approve`; `uv lock --check` per project.
- **Gate:** (1) `make setup && make test` succeeds on a clean WSL checkout. (2) Each service project installs and passes its tests **alone** (`uv sync --frozen` in that project only). (3) `docker compose --profile core up` brings all core services to healthy, and `scripts/smoke.sh` exits 0 within 180 s. (4) Each service image builds from its own lockfile. (5) CI green.
- **Recorded:** image sizes; idle RAM of the core stack measured with `docker stats` on your machine (feeds the memory budget in §3.8).
- **Deliverable:** working skeleton repository.

### Step 1 — Project 2 · Layer 1: working defect detector + demo

- **Objectives:** detect and localise defects on MVTec AD with a real model; serve it.
- **Features:** dataset adapter + split manifests; PatchCore (CPU) as the primary model; EfficientAD trained on a free cloud GPU (run bundle); supervised baseline (ResNet-18) on a leakage-free split; anomaly heatmaps; VLM description behind the capability registry (disabled until a VLM is approved, never faked); `POST /v1/inspections`; inspection page; MLflow logging in scripts.
- **Files:** `vision/data/{mvtec.py,splits.py}`, `vision/detectors/{base.py,patchcore.py,efficientad.py}`, `vision/supervised/classifier.py`, `vision/explain/{anomaly_map.py,vlm.py}`, `vision/api/routes/inspections.py`, `scripts/{train_detector,evaluate_detector}.py`, `finetune/efficientad.ipynb`, `dashboard/pages/2_Defect_Inspection.py`.
- **Dependencies:** step 0; approved download of MVTec AD and backbone weights.
- **Testing:** split-integrity tests (no image in two splits, calibration ∩ fit = ∅); detector-interface tests on tiny synthetic images; API tests (valid image, unsupported type, missing API key); reproducibility test.
- **Gate:** (1) PatchCore image and pixel AUROC logged per category for all 15 MVTec AD categories (or a declared subset with the reason in KNOWN\_ISSUES). (2) Re-running one category with the same seed reproduces image AUROC within ±0.005. (3) Per-category deviation from Anomalib’s published benchmark for the same model and config is computed; any deviation above 2 AUROC points has a written investigation in KNOWN\_ISSUES before step 2 starts. (4) API returns 200 with score + heatmap + decision for a valid image, 415 for an invalid file type, 401 without a key. (5) Inspection works in the dashboard against the dockerised service. (6) EfficientAD run bundle imported, or explicitly deferred in KNOWN\_ISSUES if no GPU quota.
- **Recorded:** AUROC tables (PatchCore, EfficientAD if available, supervised baseline); inspection latency p50/p95 on your CPU over ≥ 100 requests after warm-up.
- **Deliverable:** working detector demo; first EXPERIMENTS.md entries.

### Step 2 — Project 2 · Layers 2–3: hard benchmark + guarantee

- **Objectives:** generalisation evidence and a verified false-alarm guarantee.
- **Features:** MVTec AD 2 (test\_public) and VisA through the same pipeline; split-conformal and (α, δ) thresholds per category (§2.8); realised false-alarm and miss rates; repeated-split analysis; shift experiment; `POST /v1/calibrations`; model-evaluation page.
- **Files:** `vision/data/{mvtec_ad2.py,visa.py}`, `vision/calibration/{conformal.py,beta_bound.py}`, `scripts/{calibrate,evaluate_conformal,shift_experiment}.py`, `vision/api/routes/calibrations.py`.
- **Dependencies:** step 1; approved downloads (≥ 2 MVTec AD 2 categories, ≥ 2 VisA categories).
- **Testing:** conformal unit and property tests from §2.8; manifest data-validation tests.
- **Gate:** (1) AUROC results for ≥ 2 MVTec AD 2 and ≥ 2 VisA categories logged. (2) The code refuses a calibration set smaller than ⌈1/α⌉ − 1. (3) **Statistical validity check:** for α = 0.02 over R = 100 random calibration/test resplits per category, the mean realised false-alarm rate is ≤ α + 3 standard errors. If it fails, the cause is investigated (usually leakage or non-exchangeable splits) before moving on. (4) Shift experiment executed and reported, whatever the outcome.
- **Recorded:** mean and spread of realised false-alarm rate vs the Beta prediction; miss rate at q̂; marginal vs (α, δ) thresholds; false-alarm rate under shift.
- **Deliverable:** hard-benchmark results + guarantee plot.

### Step 3 — Project 1 · Layers 1–2: working diagnosis agent + evaluation

- **Objectives:** a real agent that retrieves, reasons over the graph, drafts 8D, uses MCP tools, and admits novelty.
- **Features:** NHTSA + AI4I ingestion; generated 8D/FMEA (DE+EN) flagged `generated`; hybrid retrieval (BM25 + dense + RRF + reranker); Neo4j graph; LLM gateway with budget ledger (Ollama by default); MCP server with ticket and data tools; LangGraph base graph; novelty threshold; top-3 causes; 8D report; hand-labelled set v1 with a labelling guide; evaluation harness; investigation page.
- **Files:** `diagnosis/ingestion/{nhtsa,ai4i,generate_8d}.py`, `diagnosis/retrieval/{bm25,dense,fusion,rerank}.py`, `diagnosis/graph/{schema.cypher,repository.py}`, `diagnosis/llm/{gateway,budget,providers/*}.py`, `diagnosis/agent/{state,graph,nodes/*}.py`, `diagnosis/novelty/threshold.py`, `diagnosis/reporting/eight_d.py`, `mcp_server/server.py`, `eval_sets/diagnosis/*`, `diagnosis/evaluation/run_eval.py`, `.github/workflows/eval.yml`.
- **Dependencies:** step 0; approved NHTSA/AI4I downloads, embedding/reranker weights and an Ollama model; paid API only if you approve it.
- **Testing:** retrieval tests on a toy corpus; Neo4j repository integration tests (testcontainers); MCP tool contract tests; agent tests with recorded responses; 8D schema validation; gateway refuses paid calls when disabled or over budget.
- **Gate:** (1) Labelled set has ≥ 50 cases with the guide and the dev / calibration / test / new-problem split committed. (2) Retrieval recall@5 and recall@10 compared for ≥ 2 embedding models, and the choice recorded in TECHNICAL\_DECISIONS. (3) The evaluation harness writes a metrics JSON with every field below, plus n per split. (4) The CI evaluation subset is deterministic (two consecutive runs give identical output). (5) LLM-as-judge vs your labels: Cohen’s kappa computed on ≥ 30 items, and the judge is used for reported metrics only if agreement is documented.
- **Recorded:** top-1/top-3 accuracy (test and new-problem splits, with 95% intervals), novelty precision/recall, tool-call accuracy, 8D completeness, cost per case, latency p50/p95.
- **Deliverable:** working diagnosis agent + evaluation report.

### Step 4 — Project 1 · Layer 3: guaranteed cause sets

- **Objectives:** “the true cause is in this set 90% of the time”, verified within the assumptions of §2.8.
- **Features:** scoring module (probabilities over the taxonomy); LAC and APS sets; (α, δ) option; versioned, frozen calibration runs; cause-set view; MAPIE cross-check.
- **Files:** `diagnosis/scoring/combine.py`, `diagnosis/conformal/{split,scores,beta_bound}.py`, `scripts/calibrate_causes.py`, `cause-set` endpoint.
- **Dependencies:** step 3 labelled set; simulator cases from step 5 for the large-sample check (run 4 and 5 in either order; the large-sample gate closes after step 5).
- **Testing:** conformal property tests; leakage test (a calibration case never retrieves itself, and calibration IDs ∩ scorer-fit IDs = ∅); our implementation agrees with MAPIE on identical inputs.
- **Gate:** (1) Coverage and average set size reported on the real test split for LAC and APS at α = 0.1, with Clopper–Pearson 95% intervals. (2) **Statistical validity check** on ≥ 500 simulator cases: mean coverage over R = 100 resplits ≥ 0.9 − 3 standard errors. (3) New-problem coverage reported and labelled “stress test, no guarantee”.
- **Recorded:** coverage, set size and empty-set rate per method and per data source.
- **Deliverable:** guaranteed cause sets.

### Step 5 — Factory simulator + scientist agent (signature moves A + B)

- **Objectives:** ground-truth evaluation and evidence-based answers.
- **Features:** SimPy line (≈ 4 stations), batches, shifts, sensors; ≥ 5 injectable fault types (tool wear, coolant/temperature drift, bad material batch, fixture misalignment, sensor calibration drift); scenario YAML; separate ground-truth ledger; case generator; scientist node with test templates, BH correction and evidence plots; evidence view and production-monitoring page.
- **Files:** `simulator/{line,machines,faults,sensors,scenarios,export,cases}.py`, `configs/simulator/scenarios/*.yaml`, `diagnosis/scientist/{templates,tests,runner}.py`, `scripts/{simulate,evaluate_scientist}.py`.
- **Dependencies:** step 3 agent; taxonomy aligned with fault types.
- **Testing:** determinism (same seed ⇒ identical Parquet file hashes); each fault changes only its intended signals; null-data and effect-data tests per template; access test (the agent’s DuckDB connection cannot read `ground_truth`).
- **Gate:** (1) Determinism and access tests pass. (2) **Statistical validity check:** on null data, each test template’s false-positive rate over 1,000 simulations is ≤ 0.05 + 3 standard errors. (3) The test scenario family (≥ 50 cases per fault type) was never used during development, which is recorded via a scenario manifest hash committed before the first evaluation run.
- **Recorded:** root-cause accuracy per fault type; share of correct causes confirmed and wrong causes rejected; template power per effect size.
- **Deliverable:** ground-truth evaluation + evidence-based answers.

### Step 6 — Cost-aware decisions (P2 signature move B)

- **Objectives:** decisions in euros with stated assumptions, without breaking the false-alarm guarantee.
- **Features:** probability calibration; guaranteed (constrained) and unconstrained three-way policies (§2.5 P2-C); cost config; sensitivity analysis; decision view.
- **Files:** `vision/decision/{calibrate_proba,policy,costs}.py`, `configs/costs/example_costs.yaml`, `scripts/evaluate_costs.py`.
- **Dependencies:** steps 1–2; your approval of the example cost values.
- **Testing:** policy unit tests on hand-computed cases; invariants (raising the review cost never increases the review share; the constrained policy never flags a part with score ≤ q̂).
- **Gate:** (1) Tests pass. (2) The calibration split for probabilities is disjoint from both the conformal calibration set and the test set (manifest test). (3) Every cost number in EXPERIMENTS.md is followed by the hash of the assumptions file it used.
- **Recorded:** expected cost per 10,000 parts for the guaranteed policy, unconstrained policy and fixed threshold; review share; realised false-alarm rate of the guaranteed policy; sensitivity over ≥ 3 cost ratios.
- **Deliverable:** results in euros.

### Step 7 — Connect the projects: end-to-end quality loop

- **Objectives:** the loop from new defect cluster to confirmed cause works automatically, on a core-only install.
- **Features:** image linker (images ↔ simulated parts, labelled as synthetic linkage); minimal discovery trigger; outbox event + worker; automatic case creation; confirmation endpoint and feedback writes; end-to-end demo script.
- **Files:** `simulator/linker.py`, `vision/discovery/trigger.py`, `common/events.py`, `diagnosis/worker/consumer.py`, `diagnosis/feedback/update.py`, `tests/e2e/test_quality_loop.py`, `scripts/demo_loop.py`.
- **Dependencies:** steps 1–6.
- **Testing:** idempotency (the same event delivered twice creates one case); contract tests for the event schema in both services; e2e test.
- **Gate:** (1) The e2e test passes against `docker compose --profile core` **with no optional extras installed**, scripted: inject fault → route images → cluster alert → exactly one case → cause set returned → confirm → Neo4j has the `CONFIRMED_CAUSE` edge and the BN count increased. (2) Dashboard pages of uninstalled features show “not installed” (UI test). (3) The demo script runs end-to-end on your machine.
- **Recorded:** end-to-end latency from first anomalous image to case opened; number of images needed before the alert.
- **Deliverable:** end-to-end quality loop (Tier A complete).

### Step 8 — Advanced features, one at a time (Tier B)

Order is chosen by value and dependency. Each sub-step ships as an optional extra behind the capability registry. Its gate always includes: the core-only CI job and the step-7 e2e test still pass, and the capability flips to `enabled` only when installed.

| Sub-step | Feature | Key files | Gate (measured) |
| --- | --- | --- | --- |
| 8a | Smart questions (P1 L4): BN, EIG question selection, posterior re-ranking | `diagnosis/bayes/{network,eig}.py`, `questions.yaml` | On ≥ 200 simulator cases: mean questions until the true cause is top-1 with posterior ≥ τ, for EIG vs random vs fixed order, with paired bootstrap 95% CI |
| 8b | Living FMEA + feedback loop (P1 L5–6) | `diagnosis/fmea/{compare,propose}.py`, `feedback/` | ≥ 20 proposals reviewed by you, precision reported; test proves graph + BN update after confirmation and that the calibration set is untouched |
| 8c | Discovery (P2 L4): UMAP + HDBSCAN, stability, alerts, labelling view | `vision/discovery/{buffer,cluster,alerts}.py` | Median detection delay and purity/ARI over ≥ 3 held-out defect types × ≥ 3 seeds |
| 8d | Few-shot + continual (P2 L5) | `vision/fewshot/{prototypes,replay}.py` | 5/10-shot accuracy mean ± std over ≥ 10 episodes; forgetting = old-class accuracy before − after |
| 8e | Edge (P2 L6): ONNX Runtime INT8 (OpenVINO = Tier C) | `vision/edge/{export,quantize,benchmark}.py` | FP32 vs INT8 p50/p95 latency over ≥ 200 runs after warm-up on recorded hardware; size; AUROC change |
| 8f | Own small model (P1 signature C) | `finetune/lora.ipynb`, `diagnosis/llm/providers/ollama.py` | Run bundle imported; same test set: quality metrics, cost per case, latency vs the API model (or vs a larger local model if no paid API is approved) |
| 8g | “How it should look” (P2 signature A) | `vision/explain/counterfactual.py`, `finetune/inpaint.ipynb` | On ≥ 50 defective images: share with repaired score < q̂ and SSIM outside mask, diffusion vs OpenCV baseline |
| 8h | Tier C: diffusion-generated synthetic defects | `finetune/`, `vision/synthetic/` | Effect on few-shot / supervised baseline with vs without synthetic data, same seeds |

Other Tier C items (Langfuse, Evidently drift report, OpenVINO, HF Spaces demo, MVTec leaderboard submission, weighted/adaptive conformal) are added only after Tier A is complete. Evidently is the first candidate because it supports the shift story.

### Step 9 — Public proof of work

- **Objectives:** make the work visible and verifiable.
- **Features:** simulator + 8D evaluation set on Hugging Face (dataset cards, licences, generated-data disclosure); 4–6 page technical report; README with demo video, architecture diagram, results table and “What I tried and why it failed”; CV bullets using measured numbers only.
- **Files:** `release/*`, `reports/technical_report.*`, `README.md`, `scripts/check_claims.py`.
- **Testing:** the release script regenerates the dataset bit-identically from its seed; `check_claims.py` verifies every number in README/EXPERIMENTS against MLflow run IDs.
- **Gate:** (1) The published dataset loads with `datasets.load_dataset` in a clean environment. (2) `check_claims.py` passes. (3) A fresh clone on a second machine or clean WSL distro runs `make demo` successfully. (4) No MVTec images in the repo or release (automated check).
- **Deliverable:** public release, report, 2-minute video.

## 6. Risk assessment (Task 6)

Impact and likelihood are judgements (H / M / L), not measurements.

### 6.1 Computational and environment risks

| Risk | Impact | Likelihood | Mitigation |
| --- | --- | --- | --- |
| No local GPU: EfficientAD, LoRA and inpainting are impractical on CPU | H | H | CPU-first design (PatchCore, DINOv2-S, 3B local LLM); GPU work isolated in `finetune/` notebooks on Colab/Kaggle; outputs exported and served on CPU |
| RAM exhaustion with Postgres + Neo4j + Ollama + PyTorch | M | H | Compose profiles; Neo4j heap cap; Ollama started only when needed; batch sizes in config |
| MVTec AD 2 is \~30 GB and high-resolution; PatchCore memory bank grows with image count and resolution | M | H | Per-category download; 256 px default (the paper’s default setting); coreset subsampling; start with 2–3 categories |
| Windows-specific friction (paths, line endings, Docker file I/O) | M | M | Work inside WSL2 filesystem; `.gitattributes` for line endings; everything runs via Make/Docker |
| CI cannot download MVTec (licence, size) | M | H | CI uses tiny synthetic fixtures for vision smoke tests; real metrics come from logged local runs with MLflow IDs |

### 6.2 Dependency risks

| Risk | Mitigation |
| --- | --- |
| Conflicts between the vision stack (torch, anomalib, openvino) and the agent stack (langgraph, LLM SDKs) | Independent uv projects with separate lockfiles and environments (§2.10); `common` has no ML dependencies; CI proves each project installs alone |
| Shared libraries drift to different versions across services | Contract tests on all shared schemas in both services + HTTP-level e2e tests |
| PyTorch CUDA wheels bloat images | Explicit CPU wheel index in `vision` and `diagnosis`; GPU only in notebooks |
| Fast-moving libraries (Anomalib, LangGraph, MCP SDK) change APIs | Pinned in lockfiles; thin adapters around each library; upgrades as deliberate, tested changes |
| Optional extras break the core install | Core-only CI job; capability registry; import rule enforced by lint |
| A chosen pretrained model is removed or gated | Model IDs, revisions and sizes in `assets.yaml`; fallback model per role |

### 6.3 Dataset risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| NHTSA recalls describe vehicle defects, not production-line causes; no root-cause labels | H | Restrict to a few component groups; map to the taxonomy; hand-label with a written guide; use the simulator for line-level causes; report real and simulated results separately |
| Only \~50–100 labelled cases → noisy accuracy and coverage estimates | H | Report confidence intervals; enlarge with simulator-generated cases (clearly separated); never extrapolate small-sample results |
| Single labeller bias | M | Labelling guide; re-label a sample after a delay to measure self-agreement; LLM-as-judge checked against your labels |
| LLM-generated 8D/FMEA content mistaken for real data | M | `provenance=generated` on every record; separate metrics; disclosed in dataset card |
| Data leakage (defect images in both train and test; calibration on test) | H | Split manifests with hashes, committed; data-validation tests that fail CI on overlap |
| MVTec AD 2 private-test ground truth is held on MVTec’s server only | M | Report on test\_public; leaderboard submission optional |
| Licences: MVTec CC BY-NC-SA (non-commercial, share-alike); images cannot be re-hosted | M | Never commit or release MVTec images; release only own simulator data and own labels; note licence in README |

### 6.4 Method and implementation risks

| Risk | Mitigation |
| --- | --- |
| Conformal guarantees silently break when exchangeability fails (drift, lighting change, new product) | Show the failure on AD 2 lighting shift; drift monitoring; recalibration per category/model version; document that guarantees are marginal |
| LLM hallucinates causes or SQL | Causes constrained to the taxonomy via structured output; SQL from parameterised templates, read-only connection |
| Circular evaluation: agent tuned on the same simulator scenarios it is tested on | Hold out whole scenario families for test; randomise parameters; never claim simulator accuracy as real-world accuracy |
| Synthetic link between MVTec images and simulated process data looks like real data | Label it as a demonstration linkage everywhere it appears |
| UMAP/HDBSCAN results unstable between runs | Fixed seeds; alerts require stability over several runs; compare against HDBSCAN on PCA features |
| Diffusion inpainting (trained on natural images) produces poor industrial textures | Classical inpainting baseline; report honestly in “what failed” |
| INT8 hurts PatchCore distance accuracy | Measure AUROC change; keep sensitive ops in FP32 if needed; report trade-off rather than hide it |
| LLM API cost overrun | Budget guard, caching, Ollama for development, recorded responses in CI, approval before paid batch jobs |
| Scope creep vs study time (courses, exams) | Tier A first; every step is a stop point that leaves a complete, presentable system |
| Secrets leaked to GitHub | `.env` git-ignored, secret scanning in pre-commit and CI |

## 7. Development strategy (Task 7)

### 7.1 Working rules

1. **One step at a time.** A step is closed only when its completion criteria are met; no new step while a critical test fails.
2. **Vertical slices.** Each step goes from data to API to dashboard, so something works end-to-end early instead of many half-built layers.
3. **Measure before claiming.** Every number in docs, README or CV comes from a script run, logged in MLflow, with its run ID in EXPERIMENTS.md. Unrun experiments are listed as “planned”, never as results.
4. **Honest status labels.** Every feature in IMPLEMENTATION\_PLAN.md is one of: planned / in progress / working (tested) / known issue. Synthetic, generated and real data are always labelled.
5. **Simplest thing that meets the spec.** New technology only when it solves a named problem; the reason goes into TECHNICAL\_DECISIONS.md.
6. **Cost approval.** No paid API batch or paid service without your go-ahead and a printed cost estimate.

### 7.2 Definition of done (per feature)

Code with type hints and docstrings · unit tests for domain logic · API test for every endpoint (happy path, validation error, auth error) · integration test where a database is involved · dashboard element calls a tested endpoint · config in YAML, not code · docs updated (ARCHITECTURE, TESTING, EXPERIMENTS, CHANGELOG) · checklist updated.

### 7.3 Testing strategy

| Level | Tooling | Runs |
| --- | --- | --- |
| Unit (domain logic: conformal, EIG, decisions, test templates) | pytest, hypothesis | Every push |
| Data validation (split overlap, schema, provenance flags) | pytest + manifests | Every push |
| API | FastAPI test client | Every push |
| Integration (Postgres, Neo4j, DuckDB, MCP) | testcontainers | Every push (services on GitHub runners) |
| Agent behaviour | Recorded LLM responses, deterministic subset | Every push (`eval.yml`) |
| ML evaluation (full) | `scripts/evaluate_*` with MLflow | Manually / before a release |
| End-to-end | docker compose + `tests/e2e` | On main and before releases |

The spec asks for evaluation on every push; that is satisfied by the deterministic subset. Full evaluations need datasets and paid calls, so they run manually and their results are logged.

### 7.4 Reproducibility

Per-project lockfiles + Docker images built from them · global seed utility used by every script · config snapshot, git SHA and package versions logged with each MLflow run · dataset manifests with checksums · `assets.yaml` pins model revisions · cloud runs only count with a complete run bundle · simulator deterministic per seed · LLM calls at temperature 0 with prompt versions recorded.

### 7.5 Git workflow

`main` always green; one feature branch per sub-step; small pull requests with CI; conventional commit messages; a tag at the end of each roadmap step (`v0.1-step1`, …) so the project can be shown at any milestone.

### 7.6 Resuming across sessions

`docs/IMPLEMENTATION_PLAN.md` holds the checklist and a short hand-off note at the end of every session: what was done, what is broken, what is next. A new session starts by reading it, then running `make test`.

## 8. Checklist before implementation begins

**A. Needed to begin Step 0 (hardware and environment)**

- [ ] CPU model and number of cores/threads
- [ ] RAM (GB)
- [ ] GPU: NVIDIA model + VRAM, other GPU, or none
- [ ] Free disk space, and on which drive
- [ ] Windows edition and version (10 or 11, build if known)
- [ ] Installed already? WSL2 + Ubuntu version, Docker Desktop, Git, VS Code
- [ ] Repository name and visibility (default: public `industrial-quality-ai` under your GitHub account, unless you choose otherwise)

**B. Environment setup (you, once — I will give exact commands after A)**

- [ ] WSL2 with Ubuntu LTS; `.wslconfig` with the limits I recommend for your RAM
- [ ] Docker Desktop with WSL2 backend; `docker run hello-world` works inside WSL
- [ ] Git configured inside WSL; SSH key added to GitHub
- [ ] `uv` installed inside WSL
- [ ] Ollama installed inside WSL (needed from step 3)

**C. Decisions needed later (not blocking Step 0)**

- [ ] Before step 1: MVTec AD download + backbone weights (approval gate); VLM choice (API = paid, or local) — the feature stays disabled until decided
- [ ] Before step 2: which MVTec AD 2 and VisA categories (I will propose based on per-category sizes)
- [ ] Before step 3: LLM backend — Ollama only, or a paid API (provider + monthly budget cap); NHTSA scope (3–5 component groups)
- [ ] Before step 6: euro cost assumptions (I will propose example values, labelled as assumptions)
- [ ] Before any cloud GPU work: Colab and/or Kaggle account
- [ ] Before step 9: Hugging Face account
- [ ] Optional: Langfuse Cloud (only if you accept sending traces to a third party)

**D. Confirmed design defaults (change any time before the step that uses them)**

- [ ] Independent uv projects + path dependency on `common` (§2.10)
- [ ] pgvector · Streamlit only · MLflow local · PatchCore on CPU, EfficientAD on free GPU
- [ ] Conformal targets: P2 α = 2% false alarms (spec); P1 90% coverage; (α, δ) variant reported alongside
- [ ] Approval thresholds from §3.9 (all datasets, models > 200 MB or not allowlisted, all paid calls, all cloud GPU runs)
- [ ] Cause taxonomy v0 drafted in Step 0 for your review

**E. Acknowledgements**

- [ ] MVTec data is non-commercial and is never committed, released or shared publicly
- [ ] Generated and simulated data are always labelled as such
- [ ] Results enter README and CV only after they are measured and traceable to a run ID

**Step 0 starts when list A is answered.** B can be done alongside the first Step 0 tasks; C and D are not blocking.
