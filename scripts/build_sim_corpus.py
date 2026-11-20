#!/usr/bin/env python
"""Build the simulator case corpus for the diagnosis agent (step 3a).

Usage (from the repository root; low priority if other jobs are running)::

    nice -n 19 uv run python scripts/build_sim_corpus.py --workers 2
    nice -n 19 uv run python scripts/build_sim_corpus.py --data-root ~/iqs-data/sim --workers 2

Stages per split: write the scenario family (YAMLs + manifest), run it, extract
symptom profiles, write labels separately. Generated data stays outside the repo;
only the small manifest (hashes, counts, timings) is written to
``eval_sets/diagnosis/sim/<corpus>.json`` for commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from iqs_diagnosis.evaluation.corpus import (
    SPLITS,
    build_split,
    counts_by_fault,
    load_corpus_config,
    write_jsonl,
)
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary
from iqs_sim.faults import FaultType
from iqs_sim.pipeline import run_family
from iqs_sim.scenarios import generate_family, write_family

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = REPO_ROOT / "configs" / "taxonomy" / "causes.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/diagnosis/sim_corpus.yaml")
    ap.add_argument(
        "--vocabulary", type=Path, default=REPO_ROOT / "configs/diagnosis/symptoms.yaml"
    )
    ap.add_argument("--data-root", type=Path, default=Path.home() / "iqs-data" / "sim")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "eval_sets/diagnosis/sim")
    args = ap.parse_args()

    cfg = load_corpus_config(args.config)
    vocab = load_vocabulary(args.vocabulary)
    out = args.data_root / cfg.name
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} is not empty; remove it or bump the corpus name", file=sys.stderr)
        return 1
    novel = {FaultType(f) for f in cfg.novel_faults}

    manifest: dict[str, object] = {
        "corpus": cfg.name,
        "config_sha256": _sha(args.config),
        "vocabulary_version": vocab.version,
        "vocabulary_sha256": _sha(args.vocabulary),
        "novel_faults": sorted(cfg.novel_faults),
        "workers": args.workers,
        "splits": {},
    }
    t_all = time.perf_counter()
    for split in SPLITS:
        sc = cfg.splits[split]
        types = tuple(f for f in FaultType if not (split == "train" and f in novel))
        t0 = time.perf_counter()
        scenarios = generate_family(
            family=sc.family,
            seed=sc.seed,
            n_per_fault=sc.n_per_fault,
            start=cfg.start,
            duration_hours=cfg.duration_hours,
            fault_types=types,
        )
        fam = write_family(scenarios, out / "families" / split)
        index = run_family(
            sorted((out / "families" / split).glob("*.yaml")),
            out / "runs" / split,
            TAXONOMY,
            workers=args.workers,
        )
        t1 = time.perf_counter()
        profiles, labels = build_split(out / "runs" / split, vocab)
        write_jsonl(out / "profiles" / f"{split}.jsonl", profiles)
        write_jsonl(out / "labels" / f"{split}.jsonl", labels)
        t2 = time.perf_counter()
        info = {
            "family": sc.family,
            "seed": sc.seed,
            "family_sha256": fam["family_sha256"],
            "n_scenarios": fam["n_scenarios"],
            "n_cases": len(profiles),
            "cases_by_fault": counts_by_fault(labels),
            "profiles_sha256": _sha(out / "profiles" / f"{split}.jsonl"),
            "labels_sha256": _sha(out / "labels" / f"{split}.jsonl"),
            "seconds_simulate": round(t1 - t0, 1),
            "seconds_profiles": round(t2 - t1, 1),
        }
        assert index["n_cases"] == len(profiles)
        manifest["splits"][split] = info  # type: ignore[index]
        print(
            f"{split}: {fam['n_scenarios']} runs, {len(profiles)} cases, "
            f"simulate {t1 - t0:.0f}s, profiles {t2 - t1:.0f}s",
            flush=True,
        )
    manifest["seconds_total"] = round(time.perf_counter() - t_all, 1)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    path = args.manifest_dir / f"{cfg.name}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"total {manifest['seconds_total']}s; manifest {path}; data {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
