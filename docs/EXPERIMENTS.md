# Experiments

Every number claimed in the README, the report or a CV comes from a row in this file, produced
by a script and logged in MLflow with its run ID (blueprint §7.1 rule 3). Experiments not yet
run are listed as **planned**, never with numbers.

## How to record a result

| Field | Content |
| --- | --- |
| ID | `EXP-<step>-<n>`, e.g. `EXP-1-01` |
| Date | ISO date of the run |
| Question | What the run answers, in one sentence |
| Command | Exact command (`scripts/...` + config path) so anyone can re-run it |
| Config | Path to the YAML in `configs/` (snapshot logged to MLflow) |
| Data | Dataset + split manifest (with checksum) |
| Seed(s) | Seeds used; repeated runs report mean ± std |
| MLflow run ID | One per run (or the parent run for a sweep) |
| Result | Metric(s) with units and the comparison baseline |
| Notes | What surprised us, what failed |

## Results

### EXP-1-01 — PatchCore on MVTec AD vs anomalib's published numbers (in progress)

| Field | Content |
| --- | --- |
| Date | 2026-11-19 |
| Question | Does our CPU PatchCore reproduce anomalib 2.7's published image / pixel AUROC per category? |
| Command | `scripts/run_patchcore_queue.sh "official:<cat> ..."` → `scripts/train_detector.py --categories <cat> --protocol official`; table: `scripts/report_patchcore.py --protocol official` |
| Config | `configs/vision/patchcore.yaml` (WRN-50-2 tv_in1k, layers 2+3, 256 px, 10 % coreset, k = 9) |
| Data | MVTec AD (archive SHA-256 `cf4313b1…`), manifests `eval_sets/vision/mvtec_ad/<cat>.json` (seed 42); official = all train/good images, official test set |
| Seed | 42 |
| Hardware | Ryzen 7 5700U, WSL2 (9.7 GiB), CPU only, 8 torch threads; other CPU work ran in parallel, so times are upper bounds |

| Category | Image AUROC | Published | Δ | Pixel AUROC | Published | Δ | Fit min | Peak RSS | MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottle | 1.000 | 1.000 | +0.000 | 0.987 | 0.984 | +0.003 | 15.8 | 2.6 GB | `c132bd4f19e8443c8c9b6673fd9c029b` |
| grid | 0.987 | 0.959 | **+0.028** | 0.979 | 0.968 | +0.011 | 22.8 | 3.1 GB | `8dc9ff1bc93849c59d826c69ea090ee3` |

Not yet run: the other 13 categories (screw was stopped mid-run), the `deploy` protocol, the
same-seed rerun and the reload check. Grid exceeds the 0.02 tolerance **upwards**; see
KNOWN_ISSUES #11. Full table: `reports/patchcore_mvtec_ad_official.md`.

## Planned

| ID | Step | Question | Status |
| --- | --- | --- | --- |
| EXP-1-01 | 1 | Image and pixel AUROC of PatchCore and EfficientAD on MVTec AD vs published numbers | in progress (2 of 15 categories; EfficientAD not run) |
| EXP-1-02 | 1 | Supervised baseline (ResNet-18) vs PatchCore on the same sup_test images | planned (code ready, not run) |
| EXP-2-01 | 2 | Detector performance on MVTec AD 2 and VisA, incl. the lighting-shift test set | planned |
| EXP-2-02 | 2 | Realised false-alarm rate of the conformal threshold vs target α over repeated splits | planned |
| EXP-3-01 | 3 | Embedding model choice by recall@k on the retrieval set | planned |
| EXP-3-02 | 3 | Diagnosis top-3 accuracy and LLM-as-judge agreement (Cohen's κ) with hand labels | planned |
| EXP-4-01 | 4 | Empirical coverage and average size of conformal cause sets (LAC vs APS) | planned |
| EXP-5-01 | 5 | Root-cause accuracy on simulator scenarios with known injected faults | planned |
| EXP-6-01 | 6 | Expected cost per 10,000 parts: cost-aware decisions vs fixed threshold, with sensitivity analysis | planned |
