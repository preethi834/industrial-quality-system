#!/usr/bin/env python
"""Fit the scorer on the fit half of v2 dev and calibrate cause sets on the other half.

Usage::

    nice -n 19 uv run python scripts/calibrate_causes.py

- dev cases are split 50/50 by SHA-256 of their case ID (salt "<corpus>/dev"): *fit* and
  *cal*. The library is the train split. The test split is not read.
- fit half: vote parameters (extended smoothing grid) and novelty threshold
  -> ``configs/diagnosis/ranker_params_v2.json``.
- cal half, cases whose primary cause is in the library (closed label space, §2.8 point 2):
  LAC and APS thresholds at alpha = 0.1, marginal and (alpha, delta = 0.1)
  -> ``configs/diagnosis/calibration/<run_id>.json`` (frozen; never edited).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.conformal.calibration import calibrate, hash_half
from iqs_diagnosis.evaluation.corpus import load_corpus_config, load_split
from iqs_diagnosis.retrieval.library import load_library
from iqs_diagnosis.scoring.combine import (
    EXTENDED_ALPHAS,
    EXTENDED_NEIGHBOURS,
    fit_scorer,
    on_grid_edge,
)
from iqs_diagnosis.scoring.ranker import CauseRanker
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/diagnosis/sim_corpus_v2.yaml"
    )
    ap.add_argument("--data-root", type=Path, default=Path.home() / "iqs-data" / "sim")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--novelty-alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_corpus_config(args.config)
    corpus = args.data_root / cfg.name
    taxonomy = load_taxonomy(REPO_ROOT / "configs/taxonomy/causes.yaml")
    vocab = load_vocabulary(REPO_ROOT / "configs/diagnosis/symptoms.yaml")
    causes = [c.code for c in taxonomy.causes]
    library = load_library(corpus / "library" / "train.jsonl")
    library_ids = {c.case_id for c in library}
    known = {c.cause_code for c in library}

    profiles, labels = load_split(corpus, "dev")
    salt = f"{cfg.name}/dev"
    halves = [hash_half(p.case_id, salt) for p in profiles]
    fit = [(p, lab) for p, lab, h in zip(profiles, labels, halves, strict=True) if h == "fit"]
    cal = [(p, lab) for p, lab, h in zip(profiles, labels, halves, strict=True) if h == "cal"]

    report = fit_scorer(
        library,
        causes,
        [p for p, _ in fit],
        [lab.cause_code for _, lab in fit],
        novelty_alpha=args.novelty_alpha,
        fitted_on={"corpus": cfg.name, "library": "train", "tuning": "dev/fit-half"},
    )
    params = report.params
    params.save(REPO_ROOT / "configs/diagnosis/ranker_params_v2.json")
    edge_a = on_grid_edge(params.alpha, EXTENDED_ALPHAS)
    edge_n = on_grid_edge(params.n_neighbours, EXTENDED_NEIGHBOURS)
    print(
        f"fit half: {len(fit)} cases; n_neighbours={params.n_neighbours}"
        f"{' (grid edge)' if edge_n else ''}, alpha={params.alpha}"
        f"{' (GRID EDGE)' if edge_a else ' (interior)'}, NLL={report.dev_nll:.4f}, "
        f"novelty threshold={params.novelty_threshold}"
    )
    nov = report.novelty
    print(
        f"fit half novelty: false-novel {nov.false_novelty_rate:.3f} (n={nov.n_known}), "
        f"detected {nov.detection_rate:.3f} (n={nov.n_novel})"
    )

    ranker = CauseRanker(library, causes, params)
    cal_known = [(p, lab) for p, lab in cal if lab.cause_code in known]
    probs = [ranker.rank(p).probabilities for p, _ in cal_known]
    run_id = f"cal_{cfg.name}_dev_a{round(args.alpha * 100):03d}"
    run = calibrate(
        run_id=run_id,
        corpus=cfg.name,
        probabilities=probs,
        truths=[lab.cause_code for _, lab in cal_known],
        calibration_ids=[p.case_id for p, _ in cal_known],
        fit_ids=[p.case_id for p, _ in fit],
        library_ids=library_ids,
        scorer=params,
        vocabulary_version=vocab.version,
        taxonomy_version=taxonomy.version,
        alpha=args.alpha,
        delta=args.delta,
        note=(
            "Simulator calibration (dx_corpus_v2 dev, cal half, known primary causes, "
            "single- and two-fault). Marginal guarantee under exchangeability with "
            "simulator cases only; not valid for real cases."
        ),
    )
    out = REPO_ROOT / "configs/diagnosis/calibration" / f"{run_id}.json"
    run.save(out)
    print(f"cal half: {len(cal)} cases, {len(cal_known)} with a library cause -> n_calibration")
    for name, t in run.thresholds.items():
        print(f"  {name:8s} qhat={t.qhat:.6f} k={t.k}/{run.n_calibration}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
