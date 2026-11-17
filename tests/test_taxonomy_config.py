"""Validates the real cause taxonomy in configs/taxonomy/causes.yaml.

Lives in the root tests/ because it checks project configuration; the
iqs_common package itself must not depend on the repository layout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from iqs_common.settings import CommonSettings
from iqs_common.taxonomy import Taxonomy, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy" / "causes.yaml"

# Pinned on purpose: adding, removing or renaming a leaf must be a deliberate
# change (bump "version", update this value) because calibration records store it.
EXPECTED_VERSION = "0.1.0"
EXPECTED_FINGERPRINT = "cd16413c56f3798a"  # pragma: allowlist secret (taxonomy hash)

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
CATCH_ALL_SEGMENTS = {"OTHER", "OTHERS", "UNKNOWN", "MISC", "GENERAL", "UNSPECIFIED"}
# Defect appearances, not causes. CONTAMINATION is deliberately absent: it can be
# a genuine material cause.
APPEARANCE_WORDS = ("SCRATCH", "CRACK", "DENT", "STAIN", "BURR", "DISCOLOR")


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_PATH)


def test_version_and_fingerprint_are_pinned(taxonomy: Taxonomy) -> None:
    assert taxonomy.version == EXPECTED_VERSION
    assert taxonomy.fingerprint == EXPECTED_FINGERPRINT


def test_flat_with_ten_leaves(taxonomy: Taxonomy) -> None:
    assert all(cause.parent is None for cause in taxonomy.causes)
    assert taxonomy.leaf_codes == taxonomy.codes
    assert len(taxonomy.leaf_codes) == 10


def test_every_leaf_has_planned_simulator_fault_and_test_template(taxonomy: Taxonomy) -> None:
    for cause in taxonomy.causes:
        assert cause.simulator_fault and SNAKE_CASE.fullmatch(cause.simulator_fault), cause.code
        assert cause.test_template and SNAKE_CASE.fullmatch(cause.test_template), cause.code
    faults = [cause.simulator_fault for cause in taxonomy.causes]
    assert len(faults) == len(set(faults)), "each simulator fault must map to exactly one cause"


def test_no_catch_all_leaf(taxonomy: Taxonomy) -> None:
    for code in taxonomy.codes:
        assert not CATCH_ALL_SEGMENTS & set(code.split(".")), code


def test_codes_name_causes_not_defect_appearances(taxonomy: Taxonomy) -> None:
    for code in taxonomy.codes:
        assert not any(word in code for word in APPEARANCE_WORDS), code


def test_names_and_descriptions_present_in_both_languages(taxonomy: Taxonomy) -> None:
    for cause in taxonomy.causes:
        assert cause.name_en.strip() and cause.name_de.strip(), cause.code
        assert cause.description.strip(), cause.code


def test_all_six_categories_are_covered(taxonomy: Taxonomy) -> None:
    assert {cause.category.value for cause in taxonomy.causes} == {
        "machine",
        "material",
        "method",
        "measurement",
        "environment",
        "personnel",
    }


def test_settings_default_path_points_to_this_file() -> None:
    default = CommonSettings.model_fields["taxonomy_path"].default
    assert (REPO_ROOT / default).resolve() == TAXONOMY_PATH
