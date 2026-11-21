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

### EXP-1-01 — PatchCore on MVTec AD vs anomalib's published numbers (done)

| Field | Content |
| --- | --- |
| Date | 2026-11-19 |
| Question | Does our CPU PatchCore reproduce anomalib 2.7's published image / pixel AUROC per category? |
| Command | `scripts/run_patchcore_queue.sh "official:<cat> ..."` → `scripts/train_detector.py --categories <cat> --protocol official`; table: `scripts/report_patchcore.py --protocol official` |
| Config | `configs/vision/patchcore.yaml` (WRN-50-2 tv_in1k, layers 2+3, 256 px, 10 % coreset, k = 9) |
| Data | MVTec AD (archive SHA-256 `cf4313b1…`), manifests `eval_sets/vision/mvtec_ad/<cat>.json` (seed 42); official = all train/good images, official test set |
| Seed | 42 |
| Hardware | Ryzen 7 5700U, WSL2 (9.7 GiB), CPU only, 8 torch threads. Bottle and grid ran on 2026-11-19 with other CPU work in parallel (times are upper bounds); the other 28 jobs ran alone in one queue, 2026-11-20 18:38 → 2026-11-21 01:05 (6 h 27 min), all rc = 0 |

`official` = fit on all train/good images (comparable with published numbers). `deploy` = fit on
the 80 % `fit` split only; the 20 % `calib` split stays unseen for the step 2 threshold, so deploy
is not a like-for-like comparison with the published table.

| Category | Official image AUROC | Published | Δ | Official pixel AUROC | Published | Δ | Deploy image AUROC | Deploy pixel AUROC | Official fit min | Official MLflow run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottle | 1.000 | 1.000 | +0.000 | 0.987 | 0.984 | +0.003 | 1.000 | 0.987 | 15.8 | `c132bd4f` |
| cable | 0.996 | 0.990 | +0.006 | 0.989 | 0.988 | +0.001 | 0.994 | 0.989 | 11.4 | `5172721b` |
| capsule | 0.994 | 0.982 | +0.012 | 0.990 | 0.988 | +0.002 | 0.969 | 0.990 | 10.3 | `33a04095` |
| carpet | 0.987 | 0.984 | +0.003 | 0.990 | 0.988 | +0.002 | 0.988 | 0.990 | 18.8 | `40e41773` |
| grid | 0.987 | 0.959 | **+0.028** | 0.979 | 0.968 | +0.011 | 0.982 | 0.978 | 22.8 | `8dc9ff1b` |
| hazelnut | 1.000 | 1.000 | +0.000 | 0.988 | 0.987 | +0.001 | 1.000 | 0.988 | 33.7 | `a8bec0d8` |
| leather | 1.000 | 1.000 | +0.000 | 0.993 | 0.991 | +0.002 | 1.000 | 0.993 | 15.2 | `910f2355` |
| metal_nut | 0.998 | 0.994 | +0.004 | 0.989 | 0.989 | −0.000 | 0.997 | 0.989 | 11.6 | `4688c6d3` |
| pill | 0.944 | 0.924 | +0.020 | 0.982 | 0.980 | +0.002 | 0.946 | 0.982 | 17.5 | `472685fe` |
| screw | 0.945 | 0.960 | −0.015 | 0.990 | 0.989 | +0.001 | 0.932 | 0.990 | 29.5 | `f82d1ac5` |
| tile | 1.000 | 1.000 | −0.000 | 0.961 | 0.961 | −0.000 | 1.000 | 0.961 | 11.6 | `216bd999` |
| toothbrush | 0.931 | 0.933 | −0.002 | 0.989 | 0.988 | +0.001 | 0.931 | 0.989 | 1.2 | `f37eaf9f` |
| transistor | 0.999 | 1.000 | −0.001 | 0.980 | 0.981 | −0.001 | 0.998 | 0.980 | 9.6 | `082892fd` |
| wood | 0.990 | 0.989 | +0.001 | 0.939 | 0.934 | +0.005 | 0.989 | 0.939 | 13.5 | `0eb004fc` |
| zipper | 0.986 | 0.982 | +0.004 | 0.986 | 0.983 | +0.003 | 0.982 | 0.986 | 12.0 | `ba7f1f2e` |
| **mean (15)** | **0.984** | 0.980 | +0.004 | **0.982** | 0.980 | +0.002 | **0.981** | **0.982** | | |

Full tables (deploy incl. MLflow runs, provisional threshold, test false-alarm and miss rates):
`reports/patchcore_mvtec_ad_official.md`, `reports/patchcore_mvtec_ad_deploy.md`.

- **Gate 3:** 14 of 15 official categories are within ±0.02. Grid (+0.028) has the written
  investigation below (KNOWN_ISSUES #11); pill (+0.020) sits at the edge (#14).
- **Deploy vs official:** 20 % fewer fit images costs 0.003 image AUROC on average. The biggest
  drops are capsule (−0.025) and screw (−0.013 → −0.028 vs published). Pixel AUROC is unchanged.
- **Gate 2 (same-seed rerun):** toothbrush official retrained into `models/patchcore_repro`
  (MLflow `aed2e864`): image AUROC 0.93056 and pixel AUROC 0.98925, both **Δ = 0.0000**
  (tolerance ±0.005).
- **Reload check:** all 30 saved models (15 × 2 protocols) re-evaluated from `model.pt` with no
  refit, `max_abs_diff = 0.0` on every compared metric (`scripts/evaluate_detector.py`).
- **Seen in the deploy table, input for step 2:** with the provisional threshold (largest calib
  score), the false-alarm rate on the official good test images is 0.625 for leather (20 of 32),
  0.333 for toothbrush and 0.321 for carpet, against 0–0.14 elsewhere. Many good test images in
  these categories score above every held-out good train image, which suggests MVTec's train-good
  and test-good images are not exchangeable there (KNOWN_ISSUES #17).

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

### EXP-1-02 — Supervised ResNet-18 vs PatchCore on the same sup_test images (done)

| Field | Content |
| --- | --- |
| Date | 2026-11-21 |
| Question | How does a supervised classifier trained on labelled defects compare with PatchCore (no defects needed) on identical images? |
| Command | `scripts/train_supervised.py --categories <all 15>` |
| Config | `configs/vision/supervised.yaml` (ResNet-18 `a1_in1k`, 224 px, 12 epochs, lr 3e-4, epoch picked on sup_val) |
| Data | Official MVTec test set re-split per category, stratified by defect type: sup_train / sup_val / sup_test (manifests, seed 42). PatchCore = official model scored on the same sup_test images |
| Seed | 42 |
| Hardware | as EXP-1-01, nothing else running; 22 min for all 15 categories, peak RSS 1.6 GB |

| Category | Supervised sup_test AUROC | PatchCore sup_test AUROC | sup_test good / defect | MLflow run |
| --- | --- | --- | --- | --- |
| bottle | 0.965 | 1.000 | 6 / 19 | `72b48652` |
| cable | 0.937 | 0.994 | 17 / 28 | `e597e7c9` |
| capsule | 0.723 | 0.991 | 7 / 33 | `c79679a3` |
| carpet | **0.982** | 0.954 | 8 / 27 | `feef426e` |
| grid | 0.559 | 0.990 | 6 / 17 | `056ea319` |
| hazelnut | 0.983 | 1.000 | 12 / 20 | `5be16274` |
| leather | 0.986 | 1.000 | 10 / 28 | `305f0c91` |
| metal_nut | 0.628 | 1.000 | 7 / 28 | `0be35da9` |
| pill | 0.896 | 0.914 | 8 / 42 | `1bd4bde4` |
| screw | **1.000** | 0.957 | 12 / 35 | `35e291ef` |
| tile | 0.942 | 0.996 | 10 / 24 | `283e4bb8` |
| toothbrush | 0.500 | 0.889 | 4 / 9 | `2461534d` |
| transistor | 0.685 | 1.000 | 18 / 12 | `1a2a9d2b` |
| wood | 1.000 | 1.000 | 6 / 17 | `01339a38` |
| zipper | 0.953 | 0.969 | 10 / 36 | `e62a3bf0` |
| **mean (15)** | **0.849** | **0.977** | | |

PatchCore is better or equal in 13 of 15 categories. The supervised model wins on carpet and screw.
The sup_* splits are small (sup_train has 21–84 images, sup_test 13–50), so the supervised numbers are noisy:
toothbrush's sup_test has 36 good/defect pairs, and 0.500 means 18 of them are ordered correctly.
Large gaps between sup_val and sup_test AUROC (metal_nut 0.96 → 0.63, transistor 0.96 → 0.69)
show the epoch choice over-fits the tiny validation split. This is the expected cost of needing
labelled defects, and the reason PatchCore is the primary model.

### EXP-1-03 — Inspection latency through the dockerised API (done)

| Field | Content |
| --- | --- |
| Date | 2026-11-21 |
| Command | `make up PROFILES=vision` → `scripts/benchmark_inspections.py --category bottle --requests 100` (5 warm-up requests) |
| Model | bottle `deploy` PatchCore (`25b9a42e`), vision container with a 2 GB memory limit, image `iqs-vision:dev` built 2026-11-19; PatchCore inference unchanged in the 2026-11-21 rebuild (KNOWN_ISSUES #16) |
| Result | client **p50 529 ms, p95 560 ms**, mean 531 ms; server-reported p50 522 ms, p95 550 ms; sequential requests, CPU only (`reports/inspection_latency.jsonl`) |
| Note | The first request for a category also loads its model: 18.8 s for bottle on a cold container, 3.7 s for screw later |

### EXP-2-01 — PatchCore on VisA and MVTec AD 2 (deploy protocol) (done)

| Field | Content |
| --- | --- |
| Date | 2026-11-21 |
| Question | Does the step-1 PatchCore pipeline carry over to harder datasets unchanged? |
| Command | `DATASET=visa scripts/run_patchcore_queue.sh "deploy:cashew deploy:pipe_fryum"`; same with `DATASET=mvtec_ad2` for vial, fruit_jelly (`nice -n 19`) |
| Config | `configs/vision/patchcore.yaml` (as step 1: WRN-50-2 `tv_in1k`, layers 2+3, 256 px, 10 % coreset, k = 9) |
| Data | `eval_sets/vision/visa/{cashew,pipe_fryum}.json` (fit 360 / calib 90 / test 150); `eval_sets/vision/mvtec_ad2/{vial,fruit_jelly}.json` (fit 233 / 210, calib 99 / 90 incl. validation, test = test_public regular 20, shift 120 / 60) |
| Seed(s) | 42 |
| MLflow run ID | cashew `fd0328e0`, pipe_fryum `e43609c4`, vial `e96e5204`, fruit_jelly `96275194` (experiments `patchcore-visa`, `patchcore-mvtec_ad2`) |

| Category | Image AUROC | Pixel AUROC | Image AUROC, all AD 2 conditions | Train time |
| --- | --- | --- | --- | --- |
| VisA cashew | 0.975 | 0.993 | – | 34 min |
| VisA pipe_fryum | 0.989 | 0.994 | – | 34 min |
| MVTec AD 2 vial (regular) | 0.960 | 0.956 | 0.889 (140 images) | 16 min |
| MVTec AD 2 fruit_jelly (regular) | 0.920 | 0.947 | 0.892 (80 images) | 12 min |

Notes: deploy protocol only (decision D3), so no comparison with published numbers (those use the
full train set). MVTec AD 2 regular-condition AUROC rests on 5 good + 15 defective images (KNOWN_ISSUES #28).

### EXP-2-02 — Realised false-alarm rate of the conformal threshold (done)

| Field | Content |
| --- | --- |
| Date | 2026-11-21 |
| Question | Is the realised false-alarm rate of the split-conformal threshold ≤ α, as §2.8 predicts? |
| Command | `scripts/calibrate.py --dataset <ds> --categories all` then `scripts/evaluate_conformal.py --dataset <ds>` for mvtec_ad, visa, mvtec_ad2 |
| Data | deploy models of all 19 categories; per-image scores in `models/patchcore/<cat>/deploy/scores.json` |
| Seed(s) | resplits seed 42, R = 100 |
| MLflow run ID | experiments `conformal-mvtec_ad` (15 runs), `conformal-visa` (cashew `dbd38ad9`, pipe_fryum `e475d236`), `conformal-mvtec_ad2` (vial `a22cc7ca`, fruit_jelly `be176118`) |
| Result | Gate 3 **passes in 19 / 19** categories: resplit mean FA within α + 3 SE and close to the Beta-binomial prediction (e.g. leather 0.0191 vs 0.0200, cashew 0.0088 vs 0.0110, vial 0.0160 vs 0.0200); resplit sd matches the predicted sd (0.015–0.108). Reports: `reports/conformal_{mvtec_ad,visa,mvtec_ad2}.md`, plots `reports/figures/conformal_*.svg` |
| Fixed split | Official calib → official good test: within the CI of α except leather 20/32, carpet 9/28 (α = 0.02) and toothbrush 4/12 (α = 0.10): non-exchangeable train/test goods (KNOWN_ISSUES #17) |
| Miss rate | At α = 0.02: 0.867 fruit_jelly, 0.670 pipe_fryum, 0.533 vial, 0.454 pill, 0.294 screw (#29) |
| (α, δ = 0.1) | infeasible at α = 0.02 for all (n ≤ 99 < 114, #30) |
| Validation-only (N1) | MVTec AD 2 with validation/good only (n = 41 / 37, α = 0.05): q̂ 2.293 vs 2.317 (vial), 2.700 = 2.700 (fruit_jelly); same fixed FA (0/5) and miss rates |

### EXP-2-03 — In-distribution threshold under lighting shift (done)

| Field | Content |
| --- | --- |
| Date | 2026-11-21 |
| Question | How far does the realised false-alarm rate move from α when capture conditions change? |
| Command | `scripts/shift_experiment.py --part all` |
| Data | real: MVTec AD 2 test_public overexposed / underexposed / shift_1–4 (5 good + 15 defective each); synthetic: good test images of the 7 MVTec AD α = 0.02 categories under brightness ×0.7 / ×1.3, contrast ×0.7 / ×1.3, warm / cool channel gains |
| Result | `reports/shift_experiment.md`. **Real:** 0/5 good false alarms in every condition (CI up to 52 %); the shift hurts ranking instead: vial underexposed image AUROC 0.720 vs 0.960 regular, fruit_jelly overexposed 0.827 vs 0.920. **Synthetic:** FA rises far above α = 0.02 in 6 of 7 categories: screw 2.4 % → 87.8 % (cool), 51.2 % (contrast ×1.3, warm); leather 62.5 % → 100 % (brightness ×1.3, contrast ×1.3); grid 4.8 % → 42.9 % (cool); pill 0 → 26.9 %; wood 0 → 15.8 %; hazelnut 0 % under all six |
| Notes | A measurement, not a target. The conformal guarantee does not survive a capture change; recalibrate after any lighting or camera change. MVTec AD 2 shifts are too small-sample (5 goods) to measure FA |

## Planned

| ID | Step | Question | Status |
| --- | --- | --- | --- |
| EXP-1-01 | 1 | Image and pixel AUROC of PatchCore and EfficientAD on MVTec AD vs published numbers | PatchCore done (15 of 15, both protocols); EfficientAD deferred (KNOWN_ISSUES #15) |
| EXP-1-02 | 1 | Supervised baseline (ResNet-18) vs PatchCore on the same sup_test images | done |
| EXP-1-03 | 1 | Inspection latency p50 / p95 through the dockerised API | done |
| EXP-2-01 | 2 | Detector performance on MVTec AD 2 and VisA, incl. the lighting-shift test set | done (2 + 2 categories, deploy) |
| EXP-2-02 | 2 | Realised false-alarm rate of the conformal threshold vs target α over repeated splits | done (19 categories) |
| EXP-2-03 | 2 | False-alarm rate of the in-distribution threshold under real and synthetic lighting shift | done |
| EXP-3-01 | 3 | Embedding model choice by recall@k on the retrieval set | planned |
| EXP-3-02 | 3 | Diagnosis top-3 accuracy and LLM-as-judge agreement (Cohen's κ) with hand labels | planned |
| EXP-4-01 | 4 | Empirical coverage and average size of conformal cause sets (LAC vs APS) | planned |
| EXP-5-01 | 5 | Root-cause accuracy on simulator scenarios with known injected faults | planned |
| EXP-6-01 | 6 | Expected cost per 10,000 parts: cost-aware decisions vs fixed threshold, with sensitivity analysis | planned |
