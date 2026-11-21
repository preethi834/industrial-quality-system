# Known issues

Limitations, open items and unexplained behaviour. Each entry: what, impact, next action.
Status at the end of step 0 (2026-11-17), updated at the end of step 1 (2026-11-21). Numbers are stable IDs: closed entries keep their number and move to the resolved list.

## Open

| # | Issue | Impact | Next action |
| --- | --- | --- | --- |
| 1 | **CI on `main` not yet confirmed.** PR #1 checks were 5/5 green on GitHub (owner, 2026-11-19); the run on `main` after the merge has not been checked. | Step 0 gate "green CI on main" is likely met but unverified. | Check the `main` run on GitHub; close this entry if green. |
| 3 | **One unexplained integration-test failure** (1 of 6 runs of `make test-integration`, 2026-11-17, right after `make down`); the output was not captured; 5 later runs, including the same sequence, passed. | Possible flakiness in container start-up. | If it recurs, keep the full log (`pytest -m integration -rA`); CI logs will capture it. |
| 4 | **Windows PostgreSQL 17 on port 5432** on the development machine. Compose publishes Postgres on **5433**. | Tools that assume 5432 from WSL reach the Windows server, not the container. | Keep `IQS_DATABASE__PORT=5433`. |
| 5 | **`uv sync` without `--all-packages` uninstalls the workspace packages** (root project is `package = false`). | Imports of `iqs_vision`, `torch`, ... break after a plain `uv sync`. | Always use `make setup` (`uv sync --all-packages --frozen`). |
| 6 | **Docker build cache ≈ 12.5 GB** after the first image builds (uv cache mount + layers). | Disk use in the WSL VM. | `docker builder prune` when disk is tight (next build re-downloads wheels). |
| 7 | **torch (CPU, 814 MB unpacked) is in both images**; diagnosis needs it via sentence-transformers. | Images 616 / 718 MB compressed. | Revisit in step 3 when the embedding model is chosen. |
| 9 | **detect-secrets 1.5.0** is the newest release but dates from May 2024. | Detector rules may lag newer token formats. | Re-evaluate (e.g. gitleaks in CI only) if it misses something. |
| 10 | **One sample of the diagnosis container at 47 % CPU** while idle, taken ~20 s after Postgres was restarted; all other samples were idle. | Probably reconnect work; not confirmed. | Re-measure with `docker stats` over a minute in step 3. |
| 11 | **Grid image AUROC 0.987 vs anomalib's published 0.959 (+0.028)**, above the ±0.02 step-1 tolerance, in our favour. **Investigated** (EXPERIMENTS EXP-1-01): the PatchCore paper reports 0.979 for the matching 10 % coreset variant (Table S1), so ours is within +0.008 of the original method; anomalib's figure is the outlier. Likely cause: backbone weights (`tv_in1k` vs `racm_in1k`), untested. | Step 1 gate 3 met (written investigation). Our grid number should not be quoted as "beating" anomalib. | Optional: one grid run with `racm_in1k` (276 MB, needs approval; skipped 2026-11-21). |
| 13 | **Coreset selection dominates training time** and grows ~quadratically with train size (bottle 888 s of 945 s fit). Measured: the 28 remaining jobs (13 official + 15 deploy) took 6 h 27 min alone on the CPU; the longest was hazelnut official (391 images, 33.7 min). | Matters for step 2: VisA / MVTec AD 2 categories with ~360 fit images need ~1 h each. | Pick small categories (step 2 D3/D4); a smaller coreset ratio only as a documented deviation. |
| 14 | **Pill image AUROC 0.944 vs anomalib's published 0.924 (+0.020)**, at the tolerance edge, in our favour. The PatchCore paper reports 0.960 for 10 % (Table S1), so ours is −0.016 vs the paper, inside tolerance; same pattern as #11 (EXPERIMENTS EXP-1-01). | None beyond #11; listed so the edge case is not missed. | None; covered by the #11 weights hypothesis. |
| 15 | **EfficientAD not trained** (step 1 gate 6): the Colab notebook and run-bundle import are built and tested (1g), but no cloud-GPU run has been made. | No EfficientAD row in EXP-1-01. | **Deferred** (gate 6 allows it): run `finetune/efficientad/train_efficientad.ipynb` on a free GPU when there is time, then `scripts/import_run_bundle.py`. Owner to confirm the deferral. |
| 16 | **Docker images `iqs-vision:dev` / `iqs-dashboard:dev` predate the last step-1 code commits** (built 2026-11-19 12:50; `34b7d3e` changed `inspections.py`, `patchcore.py`, `pipeline.py` for EfficientAD inputs). For PatchCore the code path is unchanged (`normalized_input = True`, the default), so the step 1 Docker demo and latency are valid for PatchCore. An offline rebuild (`--network=none`) failed: building the workspace packages needs the build backend from the package index. | EfficientAD models would not load in the current image. | Rebuild with network access (`make build`) after the owner approves it; re-run the smoke test. |
| 17 | **Train-good vs test-good images look non-exchangeable in some categories.** With the provisional threshold (largest of the held-out calib scores), the false-alarm rate on official good test images is 0.625 for leather (20 of 32), 0.333 toothbrush, 0.321 carpet (0–0.14 elsewhere; `reports/patchcore_mvtec_ad_deploy.md`). | The conformal guarantee assumes exchangeability (blueprint §2.8 A1). On the fixed calib → test split it may fail in these categories even if the code is correct. | Step 2: report the fixed split and the pooled resplits separately and investigate these categories (capture conditions, score distributions) before claiming the guarantee. |

## Resolved

| # | Issue | Resolution |
| --- | --- | --- |
| 12 | Step 1 training incomplete (stopped 2026-11-19 13:31). | Queue restarted 2026-11-20 18:38; all 28 jobs ended rc = 0 at 2026-11-21 01:05. Supervised baseline, same-seed rerun, reload check, reports, Docker demo with deploy models and the latency benchmark ran on 2026-11-21 (EXPERIMENTS EXP-1-01…03). |
