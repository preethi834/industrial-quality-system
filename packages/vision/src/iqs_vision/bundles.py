"""Run bundles: how results trained on a cloud GPU come back (blueprint Rev 2 §3.8).

A bundle is a zip with exactly these members:
    bundle.json   format, detector, category, protocol, git commit, resolved config,
                  manifest fingerprint, metrics measured in the cloud, environment,
                  and the SHA-256 of every other member
    model.pt      the detector file (EfficientADDetector.save format)

Import checks, in order: safe member names (no paths, no extras), checksums, format,
the manifest fingerprint equals the committed local manifest (same images, same splits),
and the git commit is known to this repository. A cloud result counts only if it
passes (and the local CPU re-evaluation reproduces its metrics).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_FORMAT = 1
MEMBERS = ("bundle.json", "model.pt")
MAX_MEMBER_BYTES = 2 * 1024**3


class BundleError(ValueError):
    """The bundle is malformed, tampered with, or does not match this repository."""


@dataclass(frozen=True)
class BundleInfo:
    detector: str
    category: str
    protocol: str
    git_commit: str
    manifest_fingerprint: str
    config: dict[str, Any]
    metrics: dict[str, float]
    environment: dict[str, str]
    raw: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_bundle(
    out: Path,
    model_file: Path,
    *,
    detector: str,
    category: str,
    protocol: str,
    git_commit: str,
    manifest_fingerprint: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    environment: dict[str, str],
) -> Path:
    model_bytes = model_file.read_bytes()
    meta = {
        "format": BUNDLE_FORMAT,
        "detector": detector,
        "category": category,
        "protocol": protocol,
        "git_commit": git_commit,
        "manifest_fingerprint": manifest_fingerprint,
        "config": config,
        "metrics": metrics,
        "environment": environment,
        "files": {"model.pt": _sha256(model_bytes)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle.json", json.dumps(meta, indent=2, sort_keys=True))
        z.writestr("model.pt", model_bytes)
    return out


def read_bundle(path: Path) -> tuple[BundleInfo, bytes]:
    """Validate structure and checksums; return the metadata and the model bytes."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BundleError(f"not a zip file: {path}") from exc
    with z:
        infos = {i.filename: i for i in z.infolist()}
        if sorted(infos) != sorted(MEMBERS):
            raise BundleError(f"bundle must contain exactly {MEMBERS}, found {sorted(infos)}")
        if any(i.file_size > MAX_MEMBER_BYTES for i in infos.values()):
            raise BundleError("bundle member too large")
        try:
            meta = json.loads(z.read("bundle.json"))
        except json.JSONDecodeError as exc:
            raise BundleError("bundle.json is not valid JSON") from exc
        model = z.read("model.pt")
    if meta.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"unsupported bundle format {meta.get('format')!r}")
    if meta.get("files", {}).get("model.pt") != _sha256(model):
        raise BundleError("model.pt checksum does not match bundle.json (corrupt or tampered)")
    for key in ("detector", "category", "protocol", "git_commit", "manifest_fingerprint"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise BundleError(f"bundle.json lacks {key!r}")
    info = BundleInfo(
        detector=meta["detector"],
        category=meta["category"],
        protocol=meta["protocol"],
        git_commit=meta["git_commit"],
        manifest_fingerprint=meta["manifest_fingerprint"],
        config=dict(meta.get("config", {})),
        metrics={k: float(v) for k, v in meta.get("metrics", {}).items()},
        environment={k: str(v) for k, v in meta.get("environment", {}).items()},
        raw=meta,
    )
    return info, model


def check_against_repo(info: BundleInfo, manifest_fingerprint: str, commit_known: bool) -> None:
    if info.manifest_fingerprint != manifest_fingerprint:
        raise BundleError(
            f"manifest fingerprint {info.manifest_fingerprint[:12]} != local "
            f"{manifest_fingerprint[:12]}: the cloud run used different images or splits"
        )
    if not commit_known:
        raise BundleError(f"git commit {info.git_commit[:12]} is not in this repository")
