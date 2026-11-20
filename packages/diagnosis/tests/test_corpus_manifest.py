"""Committed corpus manifests: fixed schema and hex SHA-256 hashes.

These files are excluded from the secret scan (their hashes look like high-entropy
secrets), so this test is what keeps anything other than the expected fields out.
"""

from __future__ import annotations

import json
import re

from .conftest import REPO_ROOT

MANIFEST_DIR = REPO_ROOT / "eval_sets" / "diagnosis" / "sim"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOP = {
    "corpus",
    "config_sha256",
    "vocabulary_version",
    "vocabulary_sha256",
    "novel_faults",
    "workers",
    "splits",
    "seconds_total",
}
SPLIT = {
    "family",
    "seed",
    "family_sha256",
    "n_scenarios",
    "n_cases",
    "cases_by_fault",
    "profiles_sha256",
    "labels_sha256",
    "seconds_simulate",
    "seconds_profiles",
}
# Added for corpora with near-limit magnitudes and two-fault scenarios (dx_corpus_v2).
SPLIT_OPTIONAL = {
    "magnitudes",
    "n_two_fault_cases",
    "pairs_family",
    "pairs_seed",
    "pairs_magnitudes",
    "pairs_family_sha256",
    "n_pair_scenarios",
}


def test_manifests_have_fixed_schema_and_hex_hashes() -> None:
    paths = sorted(p for p in MANIFEST_DIR.glob("*.json") if not p.stem.endswith("_family"))
    assert paths, "no corpus manifest committed"
    for path in paths:
        m = json.loads(path.read_text(encoding="utf-8"))
        assert set(m) == TOP, path
        assert m["corpus"] == path.stem
        for key in ("config_sha256", "vocabulary_sha256"):
            assert HEX64.match(m[key]), (path, key)
        assert set(m["splits"]) == {"train", "dev", "test"}
        for split, info in m["splits"].items():
            assert SPLIT <= set(info) <= SPLIT | SPLIT_OPTIONAL, (path, split)
            for key in ("family_sha256", "profiles_sha256", "labels_sha256"):
                assert HEX64.match(info[key]), (path, split, key)
            if "pairs_family_sha256" in info:
                assert HEX64.match(info["pairs_family_sha256"]), (path, split)
            n_runs = info["n_scenarios"] + info.get("n_pair_scenarios", 0)
            assert sum(info["cases_by_fault"].values()) == info["n_cases"] <= n_runs
        # Novel faults never appear in the case library.
        assert not set(m["novel_faults"]) & set(m["splits"]["train"]["cases_by_fault"])


FAMILY_KEYS = {
    "kind", "name", "seed", "n_per_fault", "duration_hours", "start", "magnitudes",
    "purpose", "generated_by", "n_scenarios", "family_sha256", "files",
}  # fmt: skip


def test_family_manifests_have_fixed_schema_and_hex_hashes() -> None:
    """Scenario-family manifests (committed before their first evaluation run, gate 3)."""
    paths = sorted(MANIFEST_DIR.glob("*_family.json"))
    assert paths, "no family manifest committed"
    for path in paths:
        m = json.loads(path.read_text(encoding="utf-8"))
        assert set(m) == FAMILY_KEYS, path
        assert m["kind"] == "scenario_family" and path.stem == f"{m['name']}_family"
        assert HEX64.match(m["family_sha256"])
        assert len(m["files"]) == m["n_scenarios"] >= 500
        assert all(HEX64.match(h) and f.endswith(".yaml") for f, h in m["files"].items())
