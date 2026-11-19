# Known issues

Limitations, open items and unexplained behaviour. Each entry: what, impact, next action.
Status at the end of step 0 (2026-11-17), updated in step 1. Numbers are stable IDs: closed entries keep their number and move to the resolved list.

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
| 11 | **Grid image AUROC 0.987 vs published 0.959 (+0.028)**, above the ±0.02 step-1 tolerance, in our favour. Likely causes, untested: torchvision V1 weights (`tv_in1k`) vs the weights anomalib benchmarked with, and anomalib's older benchmark settings. | Gate 3 requires a written investigation before step 2. | Compare with the PatchCore paper's grid value; to isolate the weights, run grid once with anomalib's default `racm_in1k` weights (needs approval, 276 MB); document the result. |
| 12 | **Step 1 training incomplete** (stopped 2026-11-19 13:31 by the owner): official done for bottle and grid only; screw stopped mid-run (MLflow run `dbbb4c17` marked KILLED); deploy protocol, supervised baseline, reproducibility rerun, reload check, Docker demo with deploy models and the latency benchmark not run. | Step 1 gates 1–3, 5 and the recorded metrics are open. | Restart the queue (IMPLEMENTATION_PLAN hand-off). |
| 13 | **Coreset selection dominates training time** (bottle 888 s of 945 s fit) and grows ~quadratically with train size; estimated 8–10 h for both protocols on all 15 categories on this laptop. | Long runs; CPU contention from other work slows them further. | Run overnight with nothing else on the CPU; consider a smaller coreset ratio only as a documented deviation. |
