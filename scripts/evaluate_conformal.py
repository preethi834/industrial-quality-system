#!/usr/bin/env python
"""Evaluate conformal cause sets on the v2 test split (step 4; run once for the report).

Usage::

    nice -n 19 uv run python scripts/evaluate_conformal.py

Uses the frozen calibration run (thresholds and scorer parameters), the v2 train library
and the stored symptom profiles of the test split (the same profiles the agent computes;
the full agent path is covered by the step-3d harness and the agent tests).

Reports per threshold variant (LAC, APS; marginal and (alpha, delta)): coverage with
Clopper-Pearson 95 % intervals, mean set size, set-size histogram, empty-set rate; for
single-fault, two-fault (primary covered, every injected cause covered) and held-out
new problems (stress test, no guarantee). Gate 2: R = 100 resplits on the pooled
calibration + test known cases. Simulator coverage says nothing about real cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.conformal.calibration import CalibrationRun
from iqs_diagnosis.conformal.scores import Method
from iqs_diagnosis.evaluation.conformal_eval import (
    ScoredCase,
    resplit_check,
    set_metrics,
    tie_stats,
)
from iqs_diagnosis.evaluation.corpus import load_corpus_config, load_split
from iqs_diagnosis.retrieval.library import load_library
from iqs_diagnosis.scoring.ranker import CauseRanker

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = REPO_ROOT / "configs/diagnosis/calibration/cal_dx_corpus_v2_dev_a010.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/diagnosis/sim_corpus_v2.yaml"
    )
    ap.add_argument("--data-root", type=Path, default=Path.home() / "iqs-data" / "sim")
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "reports/diagnosis/conformal_dx_corpus_v2_test.json"
    )
    args = ap.parse_args()

    cfg = load_corpus_config(args.config)
    corpus = args.data_root / cfg.name
    run = CalibrationRun.load(args.run)
    causes = [c.code for c in load_taxonomy(REPO_ROOT / "configs/taxonomy/causes.yaml").causes]
    library = load_library(corpus / "library" / "train.jsonl")
    known_causes = {c.cause_code for c in library}
    ranker = CauseRanker(library, causes, run.scorer)

    test_p, test_l = load_split(corpus, "test")
    used = set(run.calibration_case_ids) | set(run.fit_case_ids) | {c.case_id for c in library}
    leak = used & {p.case_id for p in test_p}
    if leak:
        raise SystemExit(f"error: {len(leak)} test cases were used for fit/calibration/library")

    def scored(profiles: list[Any], labels: list[Any]) -> list[ScoredCase]:
        out = []
        for p, lab in zip(profiles, labels, strict=True):
            r = ranker.rank(p)
            out.append(
                ScoredCase(
                    case_id=p.case_id,
                    probabilities=r.probabilities,
                    truth=lab.cause_code,
                    secondary=tuple(lab.secondary_causes),
                    known=lab.cause_code in known_causes,
                )
            )
        return out

    test = scored(test_p, test_l)
    known = [c for c in test if c.known]
    singles = [c for c in known if not c.secondary]
    pairs = [c for c in known if c.secondary]
    novel = [c for c in test if not c.known]

    variants: dict[str, Any] = {}
    for name, t in run.thresholds.items():
        variants[name] = {
            "method": t.method.value,
            "qhat": t.qhat,
            "k": t.k,
            "delta": t.delta,
            "known": set_metrics(known, t.method, t.qhat).model_dump(mode="json"),
            "single_fault": set_metrics(singles, t.method, t.qhat).model_dump(mode="json"),
            "two_fault": set_metrics(pairs, t.method, t.qhat, pairs=True).model_dump(mode="json"),
            "new_problems_stress_test_no_guarantee": set_metrics(
                novel, t.method, t.qhat
            ).model_dump(mode="json"),
        }

    # Gate 2: pooled calibration (dev cal half, known) + test known cases.
    dev_p, dev_l = load_split(corpus, "dev")
    cal_ids = set(run.calibration_case_ids)
    cal_pairs = [(p, lab) for p, lab in zip(dev_p, dev_l, strict=True) if p.case_id in cal_ids]
    cal_cases = scored([p for p, _ in cal_pairs], [lab for _, lab in cal_pairs])
    pool = cal_cases + known
    gate2 = {
        m.value: resplit_check(
            pool, m, n_calibration=run.n_calibration, alpha=run.alpha, repeats=args.repeats
        ).model_dump(mode="json")
        for m in Method
    }

    report = {
        "corpus": cfg.name,
        "split": "test",
        "calibration_run": run.run_id,
        "alpha": run.alpha,
        "delta": run.delta,
        "n_calibration": run.n_calibration,
        "scorer": run.scorer.model_dump(mode="json"),
        "note": "Simulator coverage is not real-world coverage. Gate 1 (real cases) is open.",
        "counts": {
            "test_cases": len(test),
            "known": len(known),
            "single_fault": len(singles),
            "two_fault": len(pairs),
            "new_problems": len(novel),
            "gate2_pool": len(pool),
        },
        "calibration_score_ties": {m: tie_stats(s) for m, s in run.calibration_scores.items()},
        "variants": variants,
        "gate2_resplits": gate2,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{cfg.name} test: {report['counts']}")
    for name, v in variants.items():
        k = v["known"]
        cov = k["coverage"]
        print(
            f"{name:8s} q={v['qhat']:.4f}  coverage {cov['value']:.3f} "
            f"[{cov['low']:.3f}, {cov['high']:.3f}] ({cov['k']}/{cov['n']})  "
            f"size {k['mean_size']:.2f}  empty {k['empty']['value']:.3f}  "
            f"hist {k['size_histogram']}"
        )
        s, p, nv = v["single_fault"], v["two_fault"], v["new_problems_stress_test_no_guarantee"]
        print(
            f"         single cov {s['coverage']['value']:.3f} size {s['mean_size']:.2f} | "
            f"two-fault cov {p['coverage']['value']:.3f} [{p['coverage']['low']:.3f}, "
            f"{p['coverage']['high']:.3f}] all-injected {p['all_injected_covered']['value']:.3f} "
            f"size {p['mean_size']:.2f} | new problems cov {nv['coverage']['value']:.3f} "
            f"empty {nv['empty']['value']:.3f} size {nv['mean_size']:.2f}"
        )
    for m, g in gate2.items():
        print(
            f"gate 2 {m}: mean coverage {g['mean_coverage']:.4f} (sd {g['sd_coverage']:.4f}, "
            f"se {g['se']:.4f}) >= {g['lower_rule']:.4f}? {g['passed']}  "
            f"mean size {g['mean_set_size']:.2f}  (pool {g['n_pool']}, R={g['repeats']})"
        )
    print(f"ties: {report['calibration_score_ties']}")
    print(f"wrote {args.out}")
    print("Simulator coverage is not real-world coverage. Gate 1 (real cases) is open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
