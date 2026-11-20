#!/usr/bin/env python
"""Evaluate the scientist agent on the sci_eval_v1 family (step 5; run once).

Usage::

    nice -n 19 uv run python scripts/evaluate_scientist.py --workers 2

Refuses to run unless the family on disk has the family_sha256 committed in
``eval_sets/diagnosis/sim/sci_eval_v1_family.json`` (gate 3: committed before the first
evaluation run). Simulates the family into ``~/iqs-data/sim/sci_eval_v1/runs`` (if not yet
there), then for every case: symptoms -> calibrated v2 ranker top 3 (agent mode) and all
10 templates (all-causes mode). Writes ``reports/diagnosis/scientist_sci_eval_v1.json``.
Simulator results are not real-plant results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.conformal.calibration import CalibrationRun
from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.evaluation.labels import load_case_label
from iqs_diagnosis.evaluation.scientist_eval import ScientistCase, summarize
from iqs_diagnosis.ingestion.simulator import iter_run_dirs, load_case
from iqs_diagnosis.retrieval.library import load_library
from iqs_diagnosis.scientist.runner import run_hypotheses, window_for_case
from iqs_diagnosis.scoring.ranker import CauseRanker
from iqs_diagnosis.symptoms import extract_profile
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary
from iqs_sim.pipeline import run_family

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "eval_sets/diagnosis/sim/sci_eval_v1_family.json"
RUN = REPO_ROOT / "configs/diagnosis/calibration/cal_dx_corpus_v2_dev_a010.json"
TAXONOMY = REPO_ROOT / "configs/taxonomy/causes.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.home() / "iqs-data/sim/sci_eval_v1")
    ap.add_argument(
        "--library",
        type=Path,
        default=Path.home() / "iqs-data/sim/dx_corpus_v2/library/train.jsonl",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--out", type=Path, default=REPO_ROOT / "reports/diagnosis/scientist_sci_eval_v1.json"
    )
    args = ap.parse_args()

    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    on_disk = json.loads((args.root / "family" / "manifest.json").read_text(encoding="utf-8"))
    if on_disk["family_sha256"] != committed["family_sha256"]:
        raise SystemExit("error: family on disk does not match the committed manifest")

    runs = args.root / "runs"
    t0 = time.perf_counter()
    if not runs.exists():
        run_family(
            sorted((args.root / "family").glob("*.yaml")), runs, TAXONOMY, workers=args.workers
        )
    t_sim = time.perf_counter() - t0

    taxonomy = load_taxonomy(TAXONOMY)
    vocab = load_vocabulary(REPO_ROOT / "configs/diagnosis/symptoms.yaml")
    templates = {c.code: c.test_template for c in taxonomy.causes if c.test_template}
    all_causes = [c.code for c in taxonomy.causes]
    run = CalibrationRun.load(RUN)
    library = load_library(args.library)
    known = {c.cause_code for c in library}
    ranker = CauseRanker(library, all_causes, run.scorer)

    t1 = time.perf_counter()
    cases: list[ScientistCase] = []
    for run_dir in iter_run_dirs(runs):
        case = load_case(run_dir)
        if case is None:
            continue
        label = load_case_label(run_dir)
        assert label is not None and case.observed_dir is not None
        top3 = [c for c, _ in ranker.rank(extract_profile(case, vocab)).top3]
        with ProcessStore(case.observed_dir) as store:
            w = window_for_case(case, store)
            rep_all = run_hypotheses(store, w, all_causes, templates, case_id=case.case_id)
            rep_agent = run_hypotheses(store, w, top3, templates, case_id=case.case_id)
        cases.append(
            ScientistCase(
                case_id=case.case_id,
                fault=label.simulator_fault,
                truth=label.cause_code,
                known=label.cause_code in known,
                agent_top3=top3,
                all_verdicts={h.cause_code: h.verdict for h in rep_all.hypotheses},
                all_p_adjusted={h.cause_code: h.p_adjusted for h in rep_all.hypotheses},
                agent_verdicts={h.cause_code: h.verdict for h in rep_agent.hypotheses},
                scientist_pick=rep_agent.data_supported_cause,
            )
        )
    t_eval = time.perf_counter() - t1
    summary = summarize(cases)
    report = {
        "family": committed["name"],
        "family_sha256": committed["family_sha256"],
        "n_scenarios": committed["n_scenarios"],
        "n_cases": len(cases),
        "library": "dx_corpus_v2 train",
        "scorer": run.run_id,
        "q": 0.05,
        "seconds_simulate": round(t_sim, 1),
        "seconds_evaluate": round(t_eval, 1),
        "note": "Simulator results; not real-plant results.",
        "summary": summary,
        "cases": [c.model_dump(mode="json") for c in cases],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cols = [
        ("correct_cause_supported", "correct-sup"),
        ("wrong_causes_rejected", "wrong-rej"),
        ("wrong_causes_supported", "wrong-sup"),
        ("agent_top1_accuracy", "agent-top1"),
        ("scientist_pick_accuracy", "sci-pick"),
        ("combined_accuracy", "combined"),
        ("templates_alone_accuracy", "tmpl-alone"),
    ]
    print(f"{len(cases)} cases / {committed['n_scenarios']} scenarios", end="; ")
    print(f"simulate {t_sim:.0f}s, evaluate {t_eval:.0f}s")
    print(f"{'fault':30s} {'n':>4s} " + " ".join(f"{lab:>11s}" for _, lab in cols))
    for f, m in summary["by_fault"].items():
        vals = " ".join(f"{m[key]['value']:11.3f}" for key, _ in cols)
        print(f"{f:30s} {m['cases']:4d} {vals}")
    a = summary["by_fault"]["ALL"]
    for key, _ in cols[:3]:
        d = a[key]
        print(f"ALL {key}: {d['value']:.3f} [{d['low']:.3f}, {d['high']:.3f}] ({d['k']}/{d['n']})")
    print(f"wrote {args.out}. Simulator results; not real-plant results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
