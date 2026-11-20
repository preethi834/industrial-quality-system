#!/usr/bin/env python
"""Evaluate the diagnosis agent on one split of the simulator corpus (step 3d).

Usage::

    nice -n 19 uv run python scripts/evaluate_diagnosis.py --split dev
    nice -n 19 uv run python scripts/evaluate_diagnosis.py --split test   # once, for the report

Writes ``reports/diagnosis/<corpus>_<split>.json`` (summary + per-case results).
Simulator accuracy is not real-world accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.evaluation.corpus import load_corpus_config
from iqs_diagnosis.evaluation.run_eval import collect_cases, evaluate_cases, summarize
from iqs_diagnosis.retrieval.library import load_library
from iqs_diagnosis.scoring.ranker import RankerParams
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/diagnosis/sim_corpus.yaml")
    ap.add_argument("--data-root", type=Path, default=Path.home() / "iqs-data" / "sim")
    ap.add_argument(
        "--params", type=Path, default=REPO_ROOT / "configs/diagnosis/ranker_params.json"
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports/diagnosis")
    ap.add_argument(
        "--library-corpus",
        help="take the case library from this corpus instead (default: the evaluated corpus)",
    )
    ap.add_argument("--tag", default="", help="suffix for the report file name")
    args = ap.parse_args()

    cfg = load_corpus_config(args.config)
    corpus = args.data_root / cfg.name
    taxonomy = load_taxonomy(REPO_ROOT / "configs/taxonomy/causes.yaml")
    # Single-fault runs live in runs/<split>, two-fault runs (if any) in runs/<split>-pairs.
    cases = []
    for runs in (corpus / "runs" / args.split, corpus / "runs" / f"{args.split}-pairs"):
        if runs.is_dir():
            cases += collect_cases(runs)
    cases.sort(key=lambda cl: cl[0].case_id)
    if len({c.case_id for c, _ in cases}) != len(cases):
        raise SystemExit("error: duplicate case ids across run directories")
    t0 = time.perf_counter()
    results = asyncio.run(
        evaluate_cases(
            cases,
            library=load_library(
                args.data_root / (args.library_corpus or cfg.name) / "library" / "train.jsonl"
            ),
            params=RankerParams.load(args.params),
            vocabulary=load_vocabulary(REPO_ROOT / "configs/diagnosis/symptoms.yaml"),
            taxonomy=taxonomy,
        )
    )
    wall = time.perf_counter() - t0
    s = summarize(results, len(taxonomy.causes))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{cfg.name}_{args.split}{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(
        json.dumps(
            {
                "corpus": cfg.name,
                "split": args.split,
                "library_corpus": args.library_corpus or cfg.name,
                "params": str(args.params.resolve().relative_to(REPO_ROOT)),
                "note": "Simulator accuracy is not real-world accuracy.",
                "wall_seconds": round(wall, 1),
                "summary": s.model_dump(mode="json"),
                "cases": [r.model_dump(mode="json") for r in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{cfg.name} / {args.split}: {s.n_cases} cases ({s.n_known} known, {s.n_novel} novel)")
    print(f"top-1 (known causes)   {s.top1_known}")
    print(f"top-3 (known causes)   {s.top3_known}")
    print(f"top-3 (all cases)      {s.top3_all}")
    print(f"false novelty (known)  {s.false_novelty}")
    print(f"novelty detected       {s.novelty_detection}")
    for name, m in s.baselines_known.items():
        print(f"baseline {name:12s}  top-1 {m['top1']}  top-3 {m['top3']}")
    rnd = s.random_expected
    print(f"baseline random        top-1 {rnd['top1']:.3f}  top-3 {rnd['top3']:.3f} (expected)")
    print(f"top-1 single-fault     {s.single_top1_known}")
    pr = s.pairs
    print(f"two-fault (n={pr.n})   top-1 primary {pr.top1_primary}")
    print(f"                       top-3 primary {pr.top3_primary}")
    print(f"                       top-3 all injected {pr.top3_all_injected}")
    print(f"                       top-1 any injected {pr.top1_any_injected}")
    print(f"tool-call accuracy     {s.tool_call_accuracy}")
    print(f"8D completeness        mean {s.completeness_mean:.3f}, full {s.completeness_full}")
    lat = s.latency
    print(f"latency to confirm     p50 {lat.p50_ms:.0f} ms, p95 {lat.p95_ms:.0f} ms", end="")
    print(f", max {lat.max_ms:.0f} ms")
    print(f"LLM cost               {s.llm_cost_usd} USD (no LLM)")
    print("per fault top-1:")
    for f, p in s.per_fault_top1.items():
        print(f"  {f:32s} {p}")
    print(f"wall {wall:.0f}s -> {out}")
    print("Simulator accuracy is not real-world accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
