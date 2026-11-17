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

None yet. Step 0 contains no experiments. Infrastructure measurements (image sizes, idle memory)
are recorded in the ADRs in `TECHNICAL_DECISIONS.md`, not here, because they are not model
results.

## Planned

| ID | Step | Question | Status |
| --- | --- | --- | --- |
| EXP-1-01 | 1 | Image and pixel AUROC of PatchCore and EfficientAD on MVTec AD vs published numbers | planned |
| EXP-1-02 | 1 | Supervised baseline (ResNet/ViT) vs unsupervised detectors on the same splits | planned |
| EXP-2-01 | 2 | Detector performance on MVTec AD 2 and VisA, incl. the lighting-shift test set | planned |
| EXP-2-02 | 2 | Realised false-alarm rate of the conformal threshold vs target α over repeated splits | planned |
| EXP-3-01 | 3 | Embedding model choice by recall@k on the retrieval set | planned |
| EXP-3-02 | 3 | Diagnosis top-3 accuracy and LLM-as-judge agreement (Cohen's κ) with hand labels | planned |
| EXP-4-01 | 4 | Empirical coverage and average size of conformal cause sets (LAC vs APS) | planned |
| EXP-5-01 | 5 | Root-cause accuracy on simulator scenarios with known injected faults | planned |
| EXP-6-01 | 6 | Expected cost per 10,000 parts: cost-aware decisions vs fixed threshold, with sensitivity analysis | planned |
