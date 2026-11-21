#!/usr/bin/env python
"""Fit the cause ranker (vote parameters + novelty threshold) on the dev split.

Usage::

    nice -n 19 uv run python scripts/fit_ranker.py            # corpus dx_corpus_v1

Reads ``profiles/`` and ``labels/`` of the train split (the case library) and the dev
split (tuning). The test split is not read. Writes the fitted parameters to
``configs/diagnosis/ranker_params.json`` and prints dev numbers, which are tuning
numbers (optimistic by construction) and not the step 3d evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.evaluation.corpus import load_corpus_config, load_split
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.scoring.ranker import CauseRanker, fit_ranker

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/diagnosis/sim_corpus.yaml")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "iqs-data" / "sim")
    ap.add_argument("--novelty-alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "configs/diagnosis/ranker_params.json")
    args = ap.parse_args()

    cfg = load_corpus_config(args.config)
    corpus = args.data_root / cfg.name
    causes = [c.code for c in load_taxonomy(REPO_ROOT / "configs/taxonomy/causes.yaml").causes]
    train_p, train_l = load_split(corpus, "train")
    dev_p, dev_l = load_split(corpus, "dev")
    library = [LibraryCase(p, lab.cause_code) for p, lab in zip(train_p, train_l, strict=True)]
    report = fit_ranker(
        library,
        causes,
        dev_p,
        [lab.cause_code for lab in dev_l],
        novelty_alpha=args.novelty_alpha,
        # Corpus name and split only: file hashes live in the corpus manifest
        # (eval_sets/diagnosis/sim/<corpus>.json).
        fitted_on={"corpus": cfg.name, "library": "train", "tuning": "dev"},
    )
    report.params.save(args.out)

    ranker = CauseRanker(library, causes, report.params)
    known = {lab.cause_code for lab in train_l}
    hits1 = hits3 = n = 0
    per_cause: dict[str, list[int]] = {}
    for p, lab in zip(dev_p, dev_l, strict=True):
        if lab.cause_code not in known:
            continue
        r = ranker.rank(p)
        top = [c for c, _ in r.top3]
        n += 1
        hits1 += top[0] == lab.cause_code
        hits3 += lab.cause_code in top
        per_cause.setdefault(lab.cause_code, [0, 0])
        per_cause[lab.cause_code][0] += top[0] == lab.cause_code
        per_cause[lab.cause_code][1] += 1
    nov = report.novelty
    print(f"library: {len(library)} cases, {len(known)} causes; dev: {len(dev_p)} cases")
    print(
        f"params: n_neighbours={report.params.n_neighbours} alpha={report.params.alpha} "
        f"novelty_threshold={report.params.novelty_threshold}"
    )
    print(f"dev (tuning data) NLL={report.dev_nll:.3f} top1={hits1}/{n} top3={hits3}/{n}")
    for cause, (ok, tot) in sorted(per_cause.items()):
        print(f"  {cause:45s} top1 {ok}/{tot}")
    print(
        f"dev novelty: false-novel {nov.false_novelty_rate:.3f} (n={nov.n_known}), "
        f"detected {nov.detection_rate:.3f} (n={nov.n_novel})"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
