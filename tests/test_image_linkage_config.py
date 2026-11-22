"""The metal_nut demo linkage shows only good images the deploy detector never trained on
(KNOWN_ISSUES #46): the step-2 split's calib goods plus test/good, none of the fit images."""

from __future__ import annotations

import json
from pathlib import Path

from iqs_sim.linker import LinkageConfig

REPO = Path(__file__).resolve().parents[1]


def test_metal_nut_goods_are_the_unseen_split_goods() -> None:
    cfg = LinkageConfig.load(REPO / "configs/simulator/image_linkage.yaml").categories["metal_nut"]
    split = json.loads((REPO / "eval_sets/vision/mvtec_ad/metal_nut.json").read_text())
    by_split: dict[str, set[str]] = {}
    for e in split["entries"]:
        if e["label"] == 0:
            by_split.setdefault(e["split"], set()).add(e["path"].removeprefix("metal_nut/"))
    assert cfg.good_images is not None
    goods = set(cfg.good_images)
    assert goods == by_split["calib"] | by_split["test"]
    assert not goods & by_split["fit"]
    assert cfg.sampling == "without_replacement"
