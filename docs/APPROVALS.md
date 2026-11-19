# Approvals

Every approved download, paid call or cloud-GPU job (blueprint Rev 2, §3.9). The allowlist with
checksums is `configs/assets.yaml`; this file records who approved what and why.

| Date | Item | Size | Licence | Purpose | Approved by |
| --- | --- | --- | --- | --- | --- |
| 2026-11-19 | MVTec AD full archive `mvtec_anomaly_detection.tar.xz` (downloaded by hand to Windows Downloads, imported with `scripts/download_dataset.py`) | ≈ 4.9 GB | CC BY-NC-SA 4.0 | Step 1 detector training / evaluation | owner (chat, step 1 plan D1) |
| 2026-11-19 | `timm/wide_resnet50_2.tv_in1k` `model.safetensors` @ `343c5fa5` | 275.8 MB | BSD-3-Clause | PatchCore backbone | owner (chat) |
| 2026-11-19 | `timm/resnet18.a1_in1k` `model.safetensors` @ `491b427b` | 46.8 MB | Apache-2.0 | Supervised baseline | owner (chat) |
| 2026-11-19 | actionlint 1.7.12 `linux_amd64` binary (checksum verified) → `~/.local/bin` | 2.4 MB | MIT | Lint GitHub workflows locally | owner (chat) |

Not approved yet (step 1g, cloud only, you start it): EfficientAD on Colab/Kaggle — needs
MVTec AD on your Drive, Imagenette2 (`imagenette2.tgz`, ≈ 1.5 GB, to verify) and anomalib's
`efficientad_pretrained_weights.zip`.
