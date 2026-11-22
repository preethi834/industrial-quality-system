"""Allowlist of external assets (datasets, model weights) and their verification.

Blueprint Rev 2, §3.9: nothing large is fetched implicitly. configs/assets.yaml lists
every asset with its source, licence, expected SHA-256 and target path; the fetch
scripts refuse to act without an explicit approval flag, and every file is verified
against its checksum before it is used. Services never download anything: they
only resolve local paths through this module and fail with AssetMissingError.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_CHUNK = 1 << 20


class AssetError(RuntimeError):
    """Base class for asset problems."""


class AssetMissingError(AssetError):
    """A required asset is not on disk. The message names the command that fetches it."""


class ChecksumMismatchError(AssetError):
    """A file's SHA-256 differs from the allowlisted value."""


class ApprovalRequiredError(AssetError):
    """A fetch was requested without the explicit approval flag."""


class _Asset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    filename: str
    size_bytes: int | None = Field(default=None, gt=0)
    sha256: Sha256
    licence: str
    target: Path
    approved: str  # ISO date of the approval recorded in docs/APPROVALS.md


class DatasetAsset(_Asset):
    source: str


class ModelAsset(_Asset):
    repo: str
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]

    @property
    def url(self) -> str:
        """Hugging Face download URL pinned to an immutable commit."""
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


class AssetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    datasets: dict[str, DatasetAsset] = {}
    models: dict[str, ModelAsset] = {}

    def model(self, name: str) -> ModelAsset:
        try:
            return self.models[name]
        except KeyError:
            raise AssetError(f"model {name!r} is not in the asset allowlist") from None

    def dataset(self, name: str) -> DatasetAsset:
        try:
            return self.datasets[name]
        except KeyError:
            raise AssetError(f"dataset {name!r} is not in the asset allowlist") from None


def load_assets(path: Path) -> AssetsConfig:
    with path.open(encoding="utf-8") as fh:
        return AssetsConfig.model_validate(yaml.safe_load(fh))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> None:
    """Raise ChecksumMismatchError unless the file has the expected SHA-256."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ChecksumMismatchError(f"{path}: sha256 {actual} != expected {expected_sha256}")


def require_approval(approved: bool, what: str) -> None:
    if not approved:
        raise ApprovalRequiredError(
            f"{what} needs explicit approval: re-run with --approve after checking "
            "source, size and licence (and record it in docs/APPROVALS.md)"
        )


def resolve_model_file(assets: AssetsConfig, name: str, root: Path) -> Path:
    """Local path of an allowlisted model file; never downloads.

    Size is checked here (cheap); the full checksum is verified by the fetch script.
    """
    asset = assets.model(name)
    path = root / asset.target
    if not path.is_file():
        raise AssetMissingError(
            f"model weights {name!r} not found at {path}; "
            f"fetch them with: uv run python scripts/fetch_models.py {name} --approve"
        )
    if asset.size_bytes is not None and path.stat().st_size != asset.size_bytes:
        raise ChecksumMismatchError(
            f"{path}: size {path.stat().st_size} != expected {asset.size_bytes}"
        )
    return path


def download_verified(url: str, dest: Path, expected_sha256: str, *, timeout_s: float = 60) -> None:
    """Stream url to dest via a .partial file; keep it only if the SHA-256 matches."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "iqs-fetch/1"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, partial.open("wb") as out:  # noqa: S310
        while chunk := resp.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != expected_sha256:
        partial.unlink()
        raise ChecksumMismatchError(f"{url}: sha256 {digest.hexdigest()} != {expected_sha256}")
    partial.replace(dest)


def extract_tar(archive: Path, target: Path, only: Sequence[str] | None = None) -> list[str]:
    """Extract a (possibly compressed) tar archive with the 'data' safety filter.

    The filter rejects absolute paths, '..' components, links leaving the target and
    device files, so an untrusted archive cannot write outside `target`.
    `only`: extract just the members whose top-level name is listed (e.g. two of VisA's
    twelve categories plus `split_csv`); every name must exist in the archive.
    Returns the top-level names extracted.
    """
    import tarfile

    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        tops = {m.name.split("/", 1)[0] for m in members}
        if only is not None:
            missing = sorted(set(only) - tops)
            if missing:
                raise AssetError(f"{archive.name}: no top-level entries {missing}")
            members = [m for m in members if m.name.split("/", 1)[0] in set(only)]
        tar.extractall(target, members=members, filter="data")
        return sorted({m.name.split("/", 1)[0] for m in members})
