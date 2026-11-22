"""Step 6b: leak check (gate 2), cost arithmetic and the evaluate_costs report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from iqs_vision.calibration.calibrate import ScoreRecord, build_calibration, save_scores
from iqs_vision.data.splits import ManifestEntry, SplitManifest
from iqs_vision.decision.costs import CostConfig, load_cost_config
from iqs_vision.decision.evaluate import (
    CategoryData,
    DecisionLeakageError,
    category_data,
    check_fit_disjoint,
    cost_per_10k,
    evaluate_category,
    image_costs,
)
from iqs_vision.decision.policy import ACCEPT, REJECT, REVIEW

REPO = Path(__file__).resolve().parents[3]
COSTS = REPO / "configs/costs/example_costs.yaml"


def _cfg() -> CostConfig:
    return load_cost_config(COSTS)[0]


# --- gate 2: probability-fit images vs calib / sup_test -------------------------------------


@pytest.mark.parametrize(
    "path", sorted((REPO / "eval_sets/vision").glob("*/*.json")), ids=lambda p: p.stem
)
def test_committed_manifests_keep_fit_set_disjoint(path: Path) -> None:
    m = SplitManifest.load(path)
    check_fit_disjoint(m)
    fit = {e.sha256 for e in m.entries if e.sup_split in ("sup_train", "sup_val")}
    assert fit, "no labelled fit images"
    assert not fit & {e.sha256 for e in m.select("calib")}
    assert not fit & {e.sha256 for e in m.select(sup_split="sup_test")}


def _e(path: str, sha: str, split: str, sup: str | None, label: int) -> ManifestEntry:
    return ManifestEntry.model_validate(
        {
            "path": path,
            "sha256": sha,
            "split": split,
            "sup_split": sup,
            "label": label,
            "defect_type": "bad" if label else "good",
            "mask": f"mask/{path}" if label else None,
        }
    )


def test_leak_by_content_is_refused() -> None:
    entries = (
        _e("calib/a.png", "h1", "calib", None, 0),
        _e("test/a_copy.png", "h1", "test", "sup_train", 0),  # same bytes as a calib image
        _e("test/b.png", "h2", "test", "sup_test", 1),
    )
    m = SplitManifest.model_construct(
        dataset="d",
        category="c",
        seed=0,
        calib_fraction=0.2,
        sup_fractions=(0.5, 0.2, 0.3),
        entries=entries,
    )
    with pytest.raises(DecisionLeakageError):
        check_fit_disjoint(m)


# --- arithmetic on a tiny table ------------------------------------------------------------


def test_image_costs_and_cost_per_10k_hand_computed() -> None:
    c = _cfg()  # C_fr 5, C_esc 200, C_rev 0.5, eps_miss 0.05, eps_fr 0.02, pi 0.01
    actions = np.array([ACCEPT, REJECT, REVIEW, ACCEPT, REJECT, REVIEW])
    labels = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(image_costs(actions, labels, c), [0, 5, 0.6, 200, 0, 10.5])
    # E(cost|good) = 5.6/3, E(cost|defect) = 210.5/3
    want = 1e4 * (0.01 * 210.5 / 3 + 0.99 * 5.6 / 3)
    assert cost_per_10k(actions, labels, c) == pytest.approx(want)


def test_test_set_defect_share_does_not_change_the_cost() -> None:
    c = _cfg()
    a1, y1 = np.array([REJECT, ACCEPT]), np.array([0, 1])
    a2, y2 = np.array([REJECT, ACCEPT, ACCEPT, ACCEPT]), np.array([0, 1, 1, 1])
    assert cost_per_10k(a1, y1, c) == pytest.approx(cost_per_10k(a2, y2, c))


def _data(seed: int = 0) -> CategoryData:
    rng = np.random.default_rng(seed)

    def mix(ng: int, nd: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.r_[rng.normal(0, 1, ng), rng.normal(3, 1, nd)],
            np.r_[np.zeros(ng, int), np.ones(nd, int)],
        )

    fs, fy = mix(30, 60)
    ts, ty = mix(15, 30)
    return CategoryData("d", "c", fs, fy, ts, ty, rng.normal(0, 1, 40), rng.normal(0, 1, 60))


def test_guaranteed_flags_exactly_the_fixed_threshold_set() -> None:
    d = _data()
    q = float(np.sort(d.calib_scores)[-2])
    row, reps = evaluate_category(d, q, 0.05, _cfg(), 200, np.random.default_rng(0))
    g, f = row["policies"]["guaranteed"], row["policies"]["fixed_conformal"]
    assert g["flag_share"] == pytest.approx(f["flag_share"])
    assert g["good"]["accept"] == pytest.approx(f["good"]["accept"])
    fa = row["false_alarm"]["guaranteed_all_test_good"]
    assert fa["flagged"] == int((d.good_test_all > q).sum())
    lo, hi = g["cost_per_10k_ci95"]
    assert lo <= g["cost_per_10k"] <= hi and reps["guaranteed"].shape == (200,)


# --- the script on a synthetic category ----------------------------------------------------


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ecosts", REPO / "scripts/evaluate_costs.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_costs_writes_report_with_hash(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    entries, recs = [], []

    def add(name: str, split: str, sup: str | None, label: int, mu: float) -> None:
        entries.append(_e(name, f"sha-{name}", split, sup, label))
        recs.append(
            ScoreRecord(
                name, f"sha-{name}", split, label, "bad" if label else "good", float(rng.normal(mu))
            )
        )

    for i in range(60):
        entries.append(_e(f"fit/{i}.png", f"sha-fit{i}", "fit", None, 0))
        add(f"calib/{i}.png", "calib", None, 0, 0)
    for sup, ng, nd in (("sup_train", 20, 40), ("sup_val", 8, 16), ("sup_test", 12, 24)):
        for i in range(ng):
            add(f"test/good/{sup}{i}.png", "test", sup, 0, 0)
        for i in range(nd):
            add(f"test/bad/{sup}{i}.png", "test", sup, 1, 3)
    m = SplitManifest(
        dataset="mvtec_ad",
        category="grid",
        seed=0,
        calib_fraction=0.2,
        sup_fractions=(0.5, 0.2, 0.3),
        entries=tuple(sorted(entries, key=lambda e: e.path)),
    )
    m.save(tmp_path / "manifests/mvtec_ad/grid.json")
    meta = {
        "dataset": "mvtec_ad",
        "category": "grid",
        "protocol": "deploy",
        "manifest_fingerprint": m.fingerprint(),
    }
    run = tmp_path / "models/grid/deploy"
    run.mkdir(parents=True)
    save_scores(run / "scores.json", meta, recs)
    (run / "calibration.json").write_text(json.dumps(build_calibration(recs, meta)))

    out = tmp_path / "r"
    rc = _script().main(
        [
            "--models",
            str(tmp_path / "models"),
            "--manifests",
            str(tmp_path / "manifests"),
            "--out-dir",
            str(out),
            "--bootstrap",
            "50",
        ]
    )
    assert rc == 0
    digest = load_cost_config(COSTS)[1]
    md = (out / "costs_vision.md").read_text()
    assert digest in md and f"sha256:{digest[:12]}" in md and "assumption" in md
    rep = json.loads((out / "costs_vision.json").read_text())
    assert rep["cost_config"]["sha256"] == digest
    assert [r["category"] for r in rep["categories"]] == ["grid"]
    assert len(rep["sensitivity"]["cost_grid"]) >= 3
    assert (out / "figures/costs_sensitivity.svg").read_text().startswith("<svg")

    # a manifest that no longer matches the scores is refused
    meta["manifest_fingerprint"] = "other"
    save_scores(run / "scores.json", meta, recs)
    with pytest.raises(ValueError, match="another manifest"):
        category_data(m, meta, recs)
