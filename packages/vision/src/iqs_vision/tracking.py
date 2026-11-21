"""MLflow logging for scripts (never imported by the API).

Local SQLite backend under mlruns/ (git-ignored); every run records the git commit,
a dirty flag, the resolved config and the split-manifest fingerprint (blueprint §7.4).
View with: uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path("mlruns")


def git_info(cwd: Path | None = None) -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": str(bool(run("status", "--porcelain"))),
    }


def environment_info() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "torch_threads": str(torch.get_num_threads()),
    }


@contextmanager
def mlflow_run(
    experiment: str,
    run_name: str,
    *,
    tracking_dir: Path = DEFAULT_DIR,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    enabled: bool = True,
) -> Iterator[Any]:
    """Start a run (or yield None when disabled); logs git + environment tags."""
    if not enabled:
        yield None
        return
    import mlflow

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        artifacts = (tracking_dir / "artifacts").resolve().as_uri()
        exp_id = mlflow.create_experiment(experiment, artifact_location=artifacts)
    else:
        exp_id = exp.experiment_id
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.set_tags({**git_info(), **environment_info(), **(tags or {})})
        if params:
            mlflow.log_params({k: _param(v) for k, v in params.items()})
        yield run


def _param(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, default=str)
