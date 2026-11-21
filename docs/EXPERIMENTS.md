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

#### Investigation: grid image AUROC above anomalib's published value (KNOWN_ISSUES #11)

Our official-protocol grid model (MLflow `8dc9ff1b`) scores image AUROC 0.987 and pixel AUROC
0.979. Anomalib's PatchCore README reports 0.959 and 0.968 for Wide ResNet-50, so the deviation
is +0.028 image (above the step 1 tolerance of ±0.02) and +0.011 pixel.

**Comparison with the original paper.** Roth et al. (CVPR 2022, arXiv 2106.08265v2, supplementary
Tables S1–S2) report grid image AUROC 0.979 for PatchCore-10 %, the variant matching our 10 %
coreset (0.982 for 25 %, 0.986 for 1 %), and grid pixel AUROC 0.987 for 25 % and 10 % and 0.986
for 1 %. Against the paper our result is **+0.008 image / −0.008 pixel**: inside the tolerance and
inside the paper's own spread across coreset ratios (0.979–0.986). Anomalib's 0.959 is 0.020 below
the paper's matching variant. Our number agrees with the original method; the anomalib figure is
the outlier.

**Known setup differences.** Paper: resize to 256 px, centre crop to 224 px, WideResNet-50 ImageNet
weights from torchvision / timm, layers 2+3, p = 3. Ours: anomalib 2.7 defaults (256 px, no centre
crop, layers 2+3, 10 % coreset, k = 9) with torchvision V1 weights (`wide_resnet50_2.tv_in1k`).
Anomalib's README states only "seed 42" and the backbone name; it does not give the weights tag,
anomalib version, image size or crop used for the benchmark. Grid is a texture, so the missing
centre crop only adds border texture; we do not expect it to explain a 0.02–0.03 gap, but this is
untested.

**Size of the effect.** The grid test set has 21 good and 57 defective images (1,197 good/defect
pairs). One misordered pair moves image AUROC by 0.0008, so +0.028 corresponds to about 33 pairs,
e.g. two good images moving across about 17 defects each. The Hanley–McNeil standard error of a
single AUROC near 0.97 on this test set is about 0.016, so grid AUROC is fragile on this test set
and small changes in how a few good images rank produce gaps of this size. Both runs use the same
test images, so the gap reflects a model or pipeline difference, not test-set sampling.

**Most likely cause (unconfirmed).** Backbone weights: anomalib loads timm's default
`wide_resnet50_2` weights (`racm_in1k` in current timm; step 1 decision record), not the
torchvision V1 weights the paper and we use. An undocumented difference in anomalib's version or
settings at benchmark time is also possible.

**Conclusion.** The deviation is in our favour and matches the original paper within tolerance. We
record it as investigated and keep the paper-matched weights. Optional confirmation (skipped on
2026-11-21, owner decision A4): retrain grid with `racm_in1k` (276 MB, one ~25 min CPU run); if
image AUROC drops to about 0.96, the weights explain the gap.

#### Note: pill image AUROC at the tolerance edge (KNOWN_ISSUES #14)

Pill (MLflow `472685fe`) scores image AUROC 0.944 vs anomalib's 0.924 (+0.020, at the edge of the
±0.02 tolerance, in our favour); pixel AUROC 0.982 vs 0.980. The PatchCore paper reports 0.960 for
PatchCore-10 % (Table S1, value supplied by the owner from the CVPR 2022 supplementary), so ours is
**−0.016 vs the paper**, inside the tolerance. The pattern matches grid: anomalib's published value
sits below the paper's (by 0.036 here) and ours lies between the two. The pill test set has 26
good and 141 defective images (3,666 pairs; one pair = 0.0003 AUROC, so +0.020 ≈ 73 pairs;
Hanley–McNeil SE ≈ 0.017). The same untested weights hypothesis as #11 applies; no extra run.

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
