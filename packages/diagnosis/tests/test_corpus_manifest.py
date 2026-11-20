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


def test_manifests_have_fixed_schema_and_hex_hashes() -> None:
    paths = sorted(MANIFEST_DIR.glob("*.json"))
    assert paths, "no corpus manifest committed"
    for path in paths:
        m = json.loads(path.read_text(encoding="utf-8"))
        assert set(m) == TOP, path
        assert m["corpus"] == path.stem
        for key in ("config_sha256", "vocabulary_sha256"):
            assert HEX64.match(m[key]), (path, key)
        assert set(m["splits"]) == {"train", "dev", "test"}
        for split, info in m["splits"].items():
            assert set(info) == SPLIT, (path, split)
            for key in ("family_sha256", "profiles_sha256", "labels_sha256"):
                assert HEX64.match(info[key]), (path, split, key)
            assert sum(info["cases_by_fault"].values()) == info["n_cases"] <= info["n_scenarios"]
        # Novel faults never appear in the case library.
        assert not set(m["novel_faults"]) & set(m["splits"]["train"]["cases_by_fault"])
