# Approvals

Every approved download, paid call or cloud-GPU job (blueprint Rev 2, §3.9). The allowlist with
checksums is `configs/assets.yaml`; this file records who approved what and why.

| Date | Item | Size | Licence | Purpose | Approved by |
| --- | --- | --- | --- | --- | --- |
| 2026-11-19 | MVTec AD full archive `mvtec_anomaly_detection.tar.xz` (downloaded by hand to Windows Downloads, imported with `scripts/download_dataset.py`) | ≈ 4.9 GB | CC BY-NC-SA 4.0 | Step 1 detector training / evaluation | owner (chat, step 1 plan D1) |
| 2026-11-19 | `timm/wide_resnet50_2.tv_in1k` `model.safetensors` @ `343c5fa5` | 275.8 MB | BSD-3-Clause | PatchCore backbone | owner (chat) |
| 2026-11-19 | `timm/resnet18.a1_in1k` `model.safetensors` @ `491b427b` | 46.8 MB | Apache-2.0 | Supervised baseline | owner (chat) |
| 2026-11-19 | actionlint 1.7.12 `linux_amd64` binary (checksum verified) → `~/.local/bin` | 2.4 MB | MIT | Lint GitHub workflows locally | owner (chat) |
| 2026-11-21 | VisA `VisA_20220922.tar` (curl from the S3 URL into `data/raw/archives/`; SHA-256 `2eb8690c…f362` = anomalib 2.7.0's, size 1,929,840,640 B = S3 Content-Length); only `cashew`, `pipe_fryum`, `split_csv`, `LICENSE-DATASET` extracted (`--only`) | 1.93 GB | CC BY 4.0 | Step 2 hard benchmark (≥ 2 VisA categories) | owner (chat, step 2 G1/G2, A1) |
| 2026-11-21 | MVTec AD 2 `vial.tar.gz` (downloaded by hand to Windows Downloads; SHA-256 `002d8671…d9ec` **measured by us**, MVTec publishes none per category) | 812 MB | CC BY-NC-SA 4.0 | Step 2 hard benchmark + real lighting-shift test | owner (chat, step 2 A2, G2, N3) |
| 2026-11-21 | MVTec AD 2 `fruit_jelly.tar.gz` (as above; SHA-256 `6ae4f379…8a2c` measured by us) | 1.29 GB | CC BY-NC-SA 4.0 | as above | owner (chat, step 2 A2, G2, N3) |
| 2026-11-21 | `scipy` as a direct dependency of `iqs-vision` (already locked via scikit-learn; `uv.lock` diff metadata-only, 2 lines, 240 packages before and after) | 0 B new | BSD-3-Clause | Beta / binomial distributions for the conformal bounds | owner (chat, step 2 A3) |

Not approved yet (step 1g, cloud only, you start it): EfficientAD on Colab/Kaggle — needs
MVTec AD on your Drive, Imagenette2 (`imagenette2.tgz`, ≈ 1.5 GB, to verify) and anomalib's
`efficientad_pretrained_weights.zip`.
