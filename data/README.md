# Data

Git-ignored working storage. Only this README is committed. Sources, licences and checksums are
allowlisted in `configs/assets.yaml`; approvals are in `docs/APPROVALS.md`.

| Dataset | Path | Source | Licence | Archive size | SHA-256 | Imported |
| --- | --- | --- | --- | --- | --- | --- |
| MVTec AD (15 categories) | `data/raw/mvtec_ad/<category>/{train,test,ground_truth}` | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) | CC BY-NC-SA 4.0 — non-commercial; never commit, release or re-host images | pending | `cf4313b1…cd3d` | pending |

Import (after downloading the archive by hand):

```bash
uv run python scripts/download_dataset.py mvtec_ad \
    --archive /mnt/c/Users/<you>/Downloads/mvtec_anomaly_detection.tar.xz --approve
```

Split manifests (paths relative to the dataset root + SHA-256 per image) live in
`eval_sets/vision/` and are committed; the images are not.
