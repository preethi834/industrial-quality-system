# Industrial AI Engineering System — Technical Blueprint

Nov 15, 2026 · Preethi Mannaru

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

1. **Two backend services, not one** — the vision stack (PyTorch, Anomalib, OpenVINO) and the agent stack (LangGraph, LLM SDKs, pgmpy) have conflicting, heavy dependency trees. Separate services with separate Docker images isolate them; a shared `common` package holds the contracts.
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
| `feedback` | Update graph, BN counts and calibration data after confirmation | Dirichlet count updates, graph upserts |
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

**P2-B Conformal threshold:** with n good calibration images and target false-alarm rate α, the threshold is the ⌈(n+1)(1−α)⌉-th smallest calibration score. Under exchangeability, a new good part exceeds it with probability at most α. Checked by repeating random calibration/test splits and reporting the spread. MVTec AD 2’s lighting-shift test set is a deliberate stress test: the guarantee should weaken under shift, which motivates drift monitoring.

**P2-C Cost-aware decisions:** calibrate scores to a defect probability p (Platt or isotonic, on a labelled split kept apart from the test set) → for each part choose the action with lowest expected cost: accept = p·C\_escape, reject = (1−p)·C\_false\_reject, review = C\_review (+ residual human error) → report expected cost per 10,000 parts vs a fixed threshold, the review share, and a sensitivity analysis over cost ratios. Euro values are stated assumptions, never presented as measured.

**P2-D Discovery:** anomalous inspection → mask from anomaly map → crop → DINOv2 embedding → distance to known defect prototypes; above a threshold the sample goes to the unknown buffer → periodic UMAP (fixed random seed) + HDBSCAN → a cluster that reaches the minimum size and stays stable across runs raises an alert and enters the labelling queue. Evaluation streams test images in time order with one held-out defect type injected from a known point, measuring detection delay (samples until alert) and purity / adjusted Rand index.

**P2-E Few-shot + continual:** prototype = mean of L2-normalised embeddings of k labelled shots; classify by cosine similarity to prototypes. Old prototypes are kept in memory (no forgetting by construction); if a trainable head is added, a replay buffer of old-class embeddings is mixed into each update. Evaluated over repeated random draws of 5 and 10 shots (mean ± std), plus old-class accuracy before vs after.

**P2-F Edge:** export (Anomalib export or `torch.onnx`) → static INT8 post-training quantisation with calibration images (ONNX Runtime `quantize_static`; OpenVINO via NNCF) → benchmark p50/p95 latency, throughput, model size, AUROC change.

**P2-G Counterfactual (“how it should look”):** dilated defect mask → diffusion inpainting → checks: anomaly score of the repaired image falls below the threshold; similarity outside the mask stays high (SSIM / LPIPS). A classical inpainting baseline (OpenCV) is compared, so the diffusion model has to earn its cost.

**P1-A Knowledge ingestion:** NHTSA recalls → clean, filter to a manageable set of component groups → map to taxonomy → LLM generates 8D + FMEA rows (cached, `provenance=generated`, a sample reviewed by hand) → chunk + embed → pgvector; entities → Neo4j.

**P1-B Bayesian network:** structure defined from the taxonomy (causes → failure modes → observable symptoms); conditional probabilities estimated from AI4I 2020 (its failure types and discretised air/process temperature, speed, torque, tool wear) and simulator data with a Bayesian estimator (Dirichlet prior). Question selection: for each unasked question Q, EIG(Q) = H(C | e) − Σ\_q P(q | e) · H(C | e, q); ask the maximum.

**P1-C Scoring + conformal:** evidence → probability vector over causes → nonconformity score (LAC: 1 − p̂(true cause); APS for adaptive sets) → threshold from calibration cases → cause set. With only \~30–40 human-labelled calibration cases the empirical coverage will vary a lot, so results are reported separately for the small real labelled set and the large simulator-generated set.

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

**Confirmation flow:** confirmed cause → graph edge `CONFIRMED_CAUSE` → BN count update → case added to the retrieval index and (if labelled) the calibration pool → FMEA proposal if the failure mode is undocumented → for vision-originated cases the cluster becomes a named defect class (few-shot prototype).

**Simulator flow:** scenario YAML (seed, line config, fault list) → SimPy run → Parquet tables + separate ground-truth ledger → cases generated from observed symptoms → used for evaluation and for the scientist agent’s tests.

## 3. Development environment (Task 3)

### 3.1 Required software

Your machine runs Windows, so the recommended setup is **WSL2 (Ubuntu) + Docker Desktop with the WSL2 backend**. Keep the repository inside the WSL filesystem (e.g. `~/code/...`), not under `/mnt/c`, because file I/O across the boundary is slow and causes path and line-ending problems.

| Tool | Status | Purpose |
| --- | --- | --- |
| WSL2 + Ubuntu LTS | Essential | Linux environment matching Docker and CI |
| Docker Desktop (WSL2 backend) + docker compose | Essential | Postgres, Neo4j, services, reproducible runs |
| Git + GitHub account | Essential | Version control, Actions CI |
| `uv` | Essential | Python versions, workspace, lockfile, fast installs |
| VS Code + WSL and Python extensions | Recommended | Editing and debugging inside WSL |
| Ollama | Essential from step 3 | Local LLM (on-prem option), also for cost-free development |
| NVIDIA driver + CUDA in WSL | Only if you have an NVIDIA GPU | GPU training locally |
| Google Colab / Kaggle notebooks | Essential if no local GPU | EfficientAD training, LoRA fine-tuning, diffusion inpainting |

### 3.2 Python version

**Python 3.11**, pinned in `.python-version` and in the Docker base images. Reason: it is the version with the least friction across the heaviest dependencies (PyTorch, Anomalib, OpenVINO, pgmpy, LangGraph) at the same time. Moving to 3.12 is a later, low-risk change once the lockfile resolves cleanly; it is not worth risk at the start. Exact package versions are not chosen in this blueprint: they are resolved and frozen in `uv.lock` in step 0, and recorded in TECHNICAL\_DECISIONS.md.

### 3.3 Dependencies by package

| Package | Essential | Optional / later |
| --- | --- | --- |
| `common` | pydantic, pydantic-settings, SQLAlchemy 2 + psycopg, pgvector (Python), structlog | — |
| `vision` | torch + torchvision (CPU wheel by default), timm, anomalib, scikit-learn, numpy, opencv-python-headless, fastapi, uvicorn, python-multipart, mlflow | umap-learn, hdbscan (step 8c) · onnx, onnxruntime (8e) · openvino, nncf (8e, optional) · pytorch-grad-cam · diffusers, accelerate (8g) · evidently · transformers (DINOv2 via HF or torch.hub) |
| `diagnosis` | langgraph, langgraph-checkpoint-postgres, LLM provider SDK (one), httpx, neo4j driver, rank-bm25 or Postgres full-text, sentence-transformers (embeddings + cross-encoder reranker), pgmpy, scipy, statsmodels, duckdb, pandas, pyarrow, fastapi, uvicorn | ragas or deepeval · langfuse · mapie (cross-check only) |
| `mcp_server` | mcp (official Python SDK) | — |
| `simulator` | simpy, numpy, pandas, pyarrow, pyyaml | datasets + huggingface\_hub (release, step 9) |
| `dashboard` | streamlit, httpx, plotly, pyvis or streamlit-agraph (graph view) | — |
| fine-tuning (runs on Colab/Kaggle, own requirements file) | — | transformers, peft, trl, bitsandbytes, datasets (step 8f) |
| dev tooling | pytest, pytest-cov, httpx test client, testcontainers, ruff, mypy, pre-commit | hypothesis (property tests), locust (load test) |

Streamlit is chosen as the single UI (the spec allows Gradio or Streamlit); one framework keeps the dashboard consistent, and Streamlit’s multipage apps fit the ten required views.

### 3.4 Hardware requirements

| Profile | CPU | RAM | Disk free | What runs |
| --- | --- | --- | --- | --- |
| Minimum | 4 cores | 16 GB | \~120 GB SSD | Everything except GPU-heavy training; run services selectively (compose profiles) |
| Recommended | 8+ cores | 32 GB | \~200 GB SSD | Full stack at once: Postgres, Neo4j, both services, Ollama 3B–7B, PatchCore |

RAM is the tightest resource: Neo4j, Postgres, an Ollama model and a PyTorch process together can exceed 16 GB. Compose profiles (`core`, `vision`, `llm`, `observability`) let you start only what a phase needs, and Neo4j’s heap is capped in config.

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
| VisA | A few GB (verify at download) | Anomalib can prepare it |
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

## 4. Repository structure (Task 4)

A `uv` workspace monorepo: one lockfile, one CI, but each package declares its own dependencies so the vision and agent stacks stay isolated in their Docker images. Working name: `industrial-quality-ai` (Python import prefix `iqs_`).

```text
industrial-quality-ai/
├── README.md                  # demo video, architecture, results table, “what failed”
├── CHANGELOG.md
├── pyproject.toml             # uv workspace root, ruff/mypy/pytest config
├── uv.lock
├── .python-version            # 3.11
├── .env.example               # every setting, no secrets
├── Makefile                   # make setup / test / eval / up / down
├── .pre-commit-config.yaml    # ruff, mypy, secret scan, large-file guard
├── .github/workflows/
│   ├── ci.yml                 # lint, types, unit + API tests on every push
│   ├── eval.yml               # deterministic evaluation subset on every push
│   └── docker.yml             # build images, compose smoke test
├── docs/
│   ├── ARCHITECTURE.md
│   ├── IMPLEMENTATION_PLAN.md # persistent checklist + session hand-off notes
│   ├── TECHNICAL_DECISIONS.md # ADR-style records
│   ├── EXPERIMENTS.md         # results, each with an MLflow run ID
│   ├── TESTING.md
│   ├── KNOWN_ISSUES.md        # limitations, unfinished features
│   └── diagrams/
├── packages/
│   ├── common/src/iqs_common/        # schemas, settings, logging, events, db clients
│   ├── vision/src/iqs_vision/
│   │   ├── data/ detectors/ supervised/ calibration/ decision/
│   │   ├── explain/ embeddings/ discovery/ fewshot/ edge/ monitoring/
│   │   ├── api/  worker/
│   │   └── tests/
│   ├── diagnosis/src/iqs_diagnosis/
│   │   ├── ingestion/ taxonomy/ retrieval/ graph/ novelty/ scoring/
│   │   ├── conformal/ bayes/ scientist/ reporting/ fmea/ feedback/
│   │   ├── llm/ agent/ evaluation/ api/ worker/
│   │   └── tests/
│   ├── mcp_server/src/iqs_mcp/       # mock QMS/MES as MCP tools
│   ├── simulator/src/iqs_sim/        # line, faults, sensors, scenarios, export, linker
│   └── dashboard/src/iqs_dashboard/  # Streamlit multipage app
├── configs/
│   ├── vision/                # per-model, per-dataset YAML
│   ├── diagnosis/             # retrieval, BN, conformal, LLM settings
│   ├── simulator/scenarios/   # fault-injection scenarios
│   ├── taxonomy/causes.yaml   # the shared cause taxonomy
│   └── costs/example_costs.yaml  # stated euro assumptions
├── scripts/                   # CLI entry points: download_*, train_*, evaluate_*, benchmark_*
├── eval_sets/
│   ├── diagnosis/             # labelled cases (JSONL), labelling guide, splits
│   └── vision/                # split manifests incl. held-out defect types
├── data/                      # git-ignored: raw/ interim/ processed/ + README with sizes and licences
├── models/                    # git-ignored: exported ONNX/OpenVINO, checkpoints
├── notebooks/                 # exploration only; never the source of truth
├── tests/
│   ├── integration/           # cross-package, real Postgres/Neo4j via testcontainers
│   ├── e2e/                   # full loop through docker compose
│   └── fixtures/              # tiny synthetic images, recorded LLM responses
├── deploy/
│   ├── docker/                # vision, diagnosis, mcp, dashboard Dockerfiles
│   └── compose/               # docker-compose.yml with profiles, init SQL, Neo4j config
├── finetune/                  # Colab/Kaggle notebooks + own requirements for LoRA and inpainting
├── reports/                   # generated figures/tables, technical report source
└── release/                   # Hugging Face dataset cards, benchmark packaging
```

| Directory | Purpose and rule |
| --- | --- |
| `packages/common` | Contracts shared by every service (event schemas, settings, DB sessions). Changing it is a cross-project change and needs integration tests. |
| `packages/vision` | All of Project 2. Domain logic (conformal, decisions, prototypes) has no I/O and is unit-tested; models sit behind interfaces. |
| `packages/diagnosis` | All of Project 1, including the LangGraph agent and the LLM gateway. Unit tests for the domain; recorded-response tests for the agent. |
| `packages/mcp_server` | Separate process exposing the mock QMS/MES as MCP tools — the agent can only change “factory systems” through it, like in a real plant. |
| `packages/simulator` | Standalone library + CLI so it can be released on its own (step 9). Deterministic per seed. |
| `packages/dashboard` | UI only. Talks to the APIs, never to databases, so every button is an API that is tested. |
| `configs/` | Every tunable value lives here, not in code. A run’s config is logged to MLflow, which makes it reproducible. |
| `scripts/` | Thin CLIs that call package code; the commands quoted in EXPERIMENTS.md. |
| `eval_sets/` | Small, version-controlled evaluation data (your labels and split manifests with checksums). Large data never goes into git. |
| `data/`, `models/` | Git-ignored working storage. `data/README.md` records sources, licences, sizes and download dates. |
| `tests/` | Cross-package tests; package-level unit tests live beside each package. |
| `deploy/` | Everything needed for `docker compose up`, with profiles to save RAM. |
| `finetune/` | GPU-only work that runs in notebooks; kept out of the service dependencies. |
| `docs/` | The seven required documents plus known issues. |
| `reports/`, `release/` | Outputs for the technical report and the public benchmark release. |

## 5. Implementation roadmap (Task 5)

The nine steps follow page 7 of the spec. I added a **Step 0 (foundation)** because none of the steps can be tested or reproduced without the repo, CI and Docker skeleton. Every step ends with a demonstrable result and an updated IMPLEMENTATION\_PLAN.md.

**Two sequencing issues in the spec, and how they are resolved:**

- Step 7 (connect the projects) needs “camera finds a new defect cluster”, but full discovery is in step 8. Resolution: step 7 builds a minimal discovery trigger (unknown-anomaly buffer + HDBSCAN on DINOv2 embeddings, no UMAP/labelling UI, no evaluation claims). Step 8c turns it into the evaluated discovery layer.
- Step 7’s loop also mentions smart questions and FMEA updates (step 8). Resolution: step 7 runs the loop with the base agent (ranked causes, conformal set, 8D, confirmation); 8a and 8b then upgrade those stages inside the already-working loop.

### Step 0 — Foundation

- **Objectives:** a repo where `make setup && make test && docker compose up` works on a clean machine.
- **Features:** uv workspace with all six packages (empty but importable), settings via `.env`, structured logging, Postgres + pgvector and Neo4j in compose, health endpoints for both services, CI (lint, types, tests), pre-commit, the seven docs + KNOWN\_ISSUES.md, initial cause taxonomy draft.
- **Files:** `pyproject.toml`, `uv.lock`, `Makefile`, `.env.example`, `deploy/compose/docker-compose.yml`, `deploy/docker/*.Dockerfile`, `.github/workflows/ci.yml`, `packages/common/src/iqs_common/{settings,logging,db}.py`, `packages/*/api/main.py`, `docs/*.md`, `configs/taxonomy/causes.yaml`.
- **Dependencies:** WSL2, Docker, uv, GitHub repo.
- **Testing:** unit test for settings loading; API tests for `/health`; compose smoke test in CI.
- **Done when:** green CI on main; both services answer `/health` in Docker; docs exist with real content.
- **Deliverable:** public (or private) GitHub repo with working skeleton.

### Step 1 — Project 2 · Layer 1: working defect detector + demo

- **Objectives:** detect and localise defects on MVTec AD with a real model; serve it.
- **Features:** dataset adapter + split manifests; PatchCore (CPU) as the primary model, EfficientAD trained on a free GPU; supervised baseline (ResNet-18) on a leakage-free split; anomaly heatmaps; VLM text description behind an interface (stub-free: if no VLM is configured the field is absent, never faked); `POST /v1/inspections`; inspection page in the dashboard; MLflow logging.
- **Files:** `vision/data/{mvtec.py,splits.py}`, `vision/detectors/{base.py,patchcore.py,efficientad.py}`, `vision/supervised/classifier.py`, `vision/explain/{anomaly_map.py,vlm.py}`, `vision/api/routes/inspections.py`, `scripts/train_detector.py`, `scripts/evaluate_detector.py`, `dashboard/pages/2_Defect_Inspection.py`.
- **Dependencies:** step 0; MVTec AD downloaded.
- **Testing:** unit tests for preprocessing and split integrity (no image in two splits); detector interface tests on tiny synthetic images; API tests (upload, invalid file, auth); evaluation script reproduces the same AUROC from the same seed.
- **Done when:** image and pixel AUROC on MVTec AD logged in MLflow and compared with published numbers; inspection works through the API and dashboard in Docker.
- **Deliverable:** working detector demo; first EXPERIMENTS.md entry.

### Step 2 — Project 2 · Layers 2–3: hard benchmark + guarantee

- **Objectives:** generalisation evidence and a verified false-alarm guarantee.
- **Features:** MVTec AD 2 (test\_public locally) and VisA through the same pipeline; split-conformal threshold per category; target vs realised false-alarm rate and miss rate; repeated-split analysis; lighting-shift stress test on AD 2.
- **Files:** `vision/data/{mvtec_ad2.py,visa.py}`, `vision/calibration/conformal.py`, `scripts/calibrate.py`, `scripts/evaluate_conformal.py`, `vision/api/routes/calibrations.py`, dashboard model-evaluation page.
- **Dependencies:** step 1; datasets downloaded.
- **Testing:** property tests for the conformal quantile (on synthetic score distributions the realised rate stays ≤ α within sampling error across many draws); data-validation tests on manifests.
- **Done when:** results table across three datasets; realised false-alarm rate reported with spread; failures (e.g. under lighting shift) documented honestly.
- **Deliverable:** hard-benchmark results + guarantee plot.

### Step 3 — Project 1 · Layers 1–2: working diagnosis agent + evaluation

- **Objectives:** a real agent that retrieves, reasons over the graph, drafts 8D, uses MCP tools, and admits novelty.
- **Features:** NHTSA + AI4I ingestion; generated 8D/FMEA (DE+EN) clearly flagged; hybrid retrieval (BM25 + dense + RRF + reranker) with the embedding model chosen by measured recall@k; Neo4j graph; LLM gateway (API + Ollama) with budget guard; MCP server with ticket tools; LangGraph base graph; novelty threshold; top-3 causes; 8D report; hand-labelled set v1 (\~50–100 cases) with labelling guide; evaluation harness; investigation page in dashboard.
- **Files:** `diagnosis/ingestion/{nhtsa.py,ai4i.py,generate_8d.py}`, `diagnosis/retrieval/{bm25.py,dense.py,fusion.py,rerank.py}`, `diagnosis/graph/{schema.cypher,repository.py}`, `diagnosis/llm/{gateway.py,providers/*.py}`, `diagnosis/agent/{state.py,graph.py,nodes/*.py}`, `diagnosis/novelty/threshold.py`, `diagnosis/reporting/eight_d.py`, `mcp_server/server.py`, `eval_sets/diagnosis/*`, `diagnosis/evaluation/run_eval.py`, `.github/workflows/eval.yml`.
- **Dependencies:** steps 0; API key + budget (or Ollama only); NHTSA files.
- **Testing:** retrieval unit tests with a toy corpus; graph repository integration tests (testcontainers Neo4j); MCP tool tests; agent tests with recorded LLM responses; 8D schema validation; LLM-as-judge agreement with your labels (Cohen’s kappa).
- **Done when:** top-1/top-3 accuracy, novelty detection rate, tool-call accuracy, 8D completeness, cost and latency per case measured on the test split; eval runs in CI on a deterministic subset.
- **Deliverable:** working diagnosis agent + evaluation report.

### Step 4 — Project 1 · Layer 3: guaranteed cause sets

- **Objectives:** “the true cause is in this set 90% of the time”, verified.
- **Features:** scoring module producing a probability vector over the taxonomy; split conformal (LAC and APS); calibration runs stored; cause-set view in the dashboard; MAPIE cross-check.
- **Files:** `diagnosis/scoring/combine.py`, `diagnosis/conformal/{split.py,scores.py}`, `scripts/calibrate_causes.py`, API `cause-set` endpoint.
- **Dependencies:** step 3 labelled set (calibration split).
- **Testing:** conformal unit/property tests; coverage evaluation with exact binomial confidence intervals.
- **Done when:** coverage and average set size reported on held-out cases (with intervals) and on held-out new problems.
- **Deliverable:** guaranteed cause sets.

### Step 5 — Factory simulator + scientist agent (signature moves A + B)

- **Objectives:** ground-truth evaluation and evidence-based answers.
- **Features:** SimPy line (e.g. 4 stations), batches, shifts, sensors; at least 5 injectable fault types (tool wear, coolant/temperature drift, bad material batch, fixture misalignment, sensor calibration drift); scenario YAML; ground-truth ledger kept separate; case generator; scientist node with test templates, BH correction and evidence plots; evidence view in dashboard; production-monitoring page.
- **Files:** `simulator/{line.py,machines.py,faults.py,sensors.py,scenarios.py,export.py,cases.py}`, `configs/simulator/scenarios/*.yaml`, `diagnosis/scientist/{templates.py,tests.py,runner.py}`, `scripts/simulate.py`, `scripts/evaluate_scientist.py`.
- **Dependencies:** step 3 agent; taxonomy aligned with fault types.
- **Testing:** simulator determinism per seed; injected faults visibly change the intended signals (and nothing else); statistical test templates validated on synthetic data with known effect / no effect (power and false-positive rate); scientist cannot read the ground-truth directory.
- **Done when:** root-cause accuracy per injected fault type; share of correct causes confirmed and wrong causes rejected by the data check.
- **Deliverable:** ground-truth evaluation + evidence-based answers.

### Step 6 — Cost-aware decisions (P2 signature move B)

- **Objectives:** decisions in euros with stated assumptions.
- **Features:** probability calibration; three-way decision rule; cost config; expected cost per 10,000 parts vs fixed threshold; review share; sensitivity analysis; decision view in the inspection page.
- **Files:** `vision/decision/{calibrate_proba.py,policy.py,costs.py}`, `configs/costs/example_costs.yaml`, `scripts/evaluate_costs.py`.
- **Dependencies:** steps 1–2.
- **Testing:** policy unit tests on hand-computed cases; invariants (raising review cost never increases review share).
- **Done when:** cost table and sensitivity plot in EXPERIMENTS.md with assumptions written next to every number.
- **Deliverable:** results in euros.

### Step 7 — Connect the projects: end-to-end quality loop

- **Objectives:** the loop from new defect cluster to confirmed cause works automatically.
- **Features:** image linker (images ↔ simulated parts); minimal discovery trigger; outbox event + worker; automatic case creation; confirmation endpoint and feedback writes; end-to-end demo script.
- **Files:** `simulator/linker.py`, `vision/discovery/trigger.py`, `common/events.py`, `diagnosis/worker/consumer.py`, `diagnosis/feedback/update.py`, `tests/e2e/test_quality_loop.py`, `scripts/demo_loop.py`.
- **Dependencies:** steps 1–6.
- **Testing:** idempotent event handling (same event twice → one case); e2e test in compose.
- **Done when:** a scripted demo shows: injected fault → defect images → cluster → case → cause set → confirmation → graph updated.
- **Deliverable:** end-to-end quality loop (Tier A complete).

### Step 8 — Advanced features, one at a time (Tier B)

Order chosen by value and dependency; each sub-step is finished, tested and documented before the next.

| Sub-step | Feature | Key files | Done when (measured) |
| --- | --- | --- | --- |
| 8a | Smart questions (P1 L4): Bayesian network, EIG question selection, posterior re-ranking | `diagnosis/bayes/{network.py,eig.py,questions.yaml}` | Questions needed to reach the right cause: EIG vs random, on simulator cases |
| 8b | Living FMEA + feedback loop (P1 L5–6) | `diagnosis/fmea/{compare.py,propose.py}`, `feedback/` | FMEA proposal precision on your review; probabilities and graph change after confirmation (tested) |
| 8c | Discovery (P2 L4): UMAP + HDBSCAN, stability, alerts, labelling view | `vision/discovery/{buffer.py,cluster.py,alerts.py}`, dashboard labelling page | Detection delay and purity on held-out defect types |
| 8d | Few-shot + continual (P2 L5) | `vision/fewshot/{prototypes.py,replay.py}` | 5/10-shot accuracy and forgetting, mean ± std over repeats |
| 8e | Edge (P2 L6): ONNX Runtime INT8 (OpenVINO as Tier C add-on) | `vision/edge/{export.py,quantize.py,benchmark.py}` | FP32 vs INT8 latency (p50/p95), size, AUROC change on CPU |
| 8f | Own small model (P1 signature C) | `finetune/lora_*.ipynb`, `diagnosis/llm/providers/ollama.py` | Quality, cost and latency vs the API model on the same test set |
| 8g | “How it should look” (P2 signature A) | `vision/explain/counterfactual.py`, `finetune/inpaint_*.ipynb` | Repaired-image anomaly score below threshold + similarity outside mask, vs classical baseline |
| 8h | Tier C: diffusion-generated synthetic defects | `vision/synthetic/` | Effect on few-shot / supervised baseline, if any |

Other Tier C items (Langfuse, Evidently drift report, OpenVINO, HF Spaces demo, MVTec leaderboard submission) are slotted in where they fit, never before Tier A is complete. Evidently is cheap to add after step 2 and supports the shift story, so it is the first Tier C candidate.

### Step 9 — Public proof of work

- **Objectives:** make the work visible and verifiable.
- **Features:** simulator + 8D evaluation set on Hugging Face (dataset cards, licences, generated-data disclosure); 4–6 page technical report; README with demo video, architecture diagram, results table and “What I tried and why it failed”; CV bullets filled only with measured numbers.
- **Files:** `release/*`, `reports/technical_report.*`, `README.md`.
- **Testing:** release script reproduces the published dataset from a seed; every number in README traced to an MLflow run.
- **Done when:** a stranger can clone, run `docker compose up`, and reproduce a headline result.
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
| Conflicts between the vision stack (torch, anomalib, openvino) and the agent stack (langgraph, LLM SDKs) | Separate packages and Docker images; one `uv.lock` resolved early in step 0 so conflicts surface on day one |
| PyTorch CUDA wheels bloat images | CPU wheel index by default; GPU only in notebooks |
| Fast-moving libraries (Anomalib, LangGraph, MCP SDK) change APIs | Pin versions in the lockfile; thin adapters around each library; upgrades as deliberate, tested changes |
| A chosen pretrained model is removed or gated on Hugging Face | Model IDs in config; record checksums; keep a fallback model per role |

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

Lockfile + Docker images · global seed utility used by every script · config snapshot and git commit logged with each MLflow run · dataset manifests with checksums · simulator deterministic per seed · LLM calls at temperature 0 with prompt versions recorded.

### 7.5 Git workflow

`main` always green; one feature branch per sub-step; small pull requests with CI; conventional commit messages; a tag at the end of each roadmap step (`v0.1-step1`, …) so the project can be shown at any milestone.

### 7.6 Resuming across sessions

`docs/IMPLEMENTATION_PLAN.md` holds the checklist and a short hand-off note at the end of every session: what was done, what is broken, what is next. A new session starts by reading it, then running `make test`.

## 8. Checklist before implementation begins

**Information I need from you**

- [ ] Hardware: CPU model/cores, RAM, NVIDIA GPU (model + VRAM) or none, free disk space
- [ ] Windows version, and whether WSL2 and Docker Desktop are already installed
- [ ] GitHub repository name and visibility (public recommended for the portfolio)
- [ ] LLM API provider (Anthropic, OpenAI or Azure OpenAI) and a monthly budget cap — or “Ollama only until step 8f”
- [ ] VLM choice for defect descriptions: API model (paid) or local (slow on CPU)

**Environment setup (you, once)**

- [ ] WSL2 with Ubuntu LTS installed
- [ ] Docker Desktop with WSL2 backend running; `docker run hello-world` works inside WSL
- [ ] Git configured inside WSL; SSH key added to GitHub
- [ ] `uv` installed inside WSL
- [ ] Ollama installed (needed from step 3)
- [ ] VS Code with WSL + Python extensions (recommended)

**Accounts and data access**

- [ ] MVTec registration; download MVTec AD (step 1) and 2–3 MVTec AD 2 categories (step 2)
- [ ] VisA download (step 2)
- [ ] NHTSA recalls + complaints flat files and AI4I 2020 (step 3)
- [ ] Google Colab and/or Kaggle account for GPU work
- [ ] Hugging Face account (model downloads now, release in step 9)
- [ ] LLM API key stored only in `.env`, with spending limit set in the provider console
- [ ] Optional: Langfuse Cloud account (only if you accept sending traces to a third party)

**Decisions to confirm (my recommendation in brackets)**

- [ ] Vector store \[pgvector — one database fewer\]
- [ ] UI framework \[Streamlit only\]
- [ ] Experiment tracking \[MLflow local\]
- [ ] Primary anomaly model \[PatchCore on CPU; EfficientAD on free GPU\]
- [ ] MVTec AD 2 starting categories \[2–3, chosen after checking per-category sizes\]
- [ ] Conformal targets \[P2: α = 2% false alarms, as in the spec; P1: 90% coverage\]
- [ ] Euro cost assumptions for the decision layer \[I will propose example values, clearly labelled as assumptions, for your approval\]
- [ ] NHTSA scope \[3–5 component groups\]
- [ ] Cause taxonomy v0 \[drafted in step 0 from simulator fault types + AI4I failure types + NHTSA component groups, for your review\]

**Acknowledgements**

- [ ] MVTec data is non-commercial and is never committed or re-hosted
- [ ] Generated and simulated data are always labelled as such
- [ ] Results enter the README and CV only after they are measured

**Step 0 starts when** the hardware information, the GitHub repository and the LLM/budget decision are in; everything else can be completed in parallel with steps 0–2.
