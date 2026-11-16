from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from iqs_common.taxonomy import Taxonomy, load_taxonomy


def _cause(code: str, category: str, parent: str | None = None) -> dict[str, Any]:
    cause: dict[str, Any] = {
        "code": code,
        "name_en": code.title(),
        "name_de": code.title(),
        "category": category,
    }
    if parent is not None:
        cause["parent"] = parent
    return cause


def _valid() -> dict[str, Any]:
    return {
        "version": "test-1",
        "causes": [
            _cause("MACHINE", "machine"),
            _cause("MACHINE.TOOL_WEAR", "machine", "MACHINE"),
            _cause("MACHINE.FIXTURE_MISALIGNMENT", "machine", "MACHINE"),
            _cause("MATERIAL.BAD_BATCH", "material"),
        ],
    }


def _with(*extra: dict[str, Any]) -> dict[str, Any]:
    data = _valid()
    data["causes"].extend(extra)
    return data


# --- loading and lookups -----------------------------------------------------------


def test_load_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "causes.yaml"
    path.write_text(yaml.safe_dump(_valid()), encoding="utf-8")
    taxonomy = load_taxonomy(path)
    assert taxonomy.version == "test-1"
    assert len(taxonomy.causes) == 4


def test_get_contains_and_children() -> None:
    taxonomy = Taxonomy.model_validate(_valid())
    assert taxonomy.get("MACHINE.TOOL_WEAR").category == "machine"
    assert "MATERIAL.BAD_BATCH" in taxonomy
    assert "MACHINE.NOPE" not in taxonomy
    assert {c.code for c in taxonomy.children("MACHINE")} == {
        "MACHINE.TOOL_WEAR",
        "MACHINE.FIXTURE_MISALIGNMENT",
    }
    with pytest.raises(KeyError, match="unknown cause code"):
        taxonomy.get("MACHINE.NOPE")
    with pytest.raises(KeyError):
        taxonomy.children("MACHINE.NOPE")


# --- leaves, labels, fingerprint ---------------------------------------------------


def test_leaf_codes_exclude_parents() -> None:
    taxonomy = Taxonomy.model_validate(_valid())
    assert taxonomy.leaf_codes == (
        "MACHINE.FIXTURE_MISALIGNMENT",
        "MACHINE.TOOL_WEAR",
        "MATERIAL.BAD_BATCH",
    )
    assert "MACHINE" in taxonomy.codes
    assert "MACHINE" not in taxonomy.leaf_codes


def test_label_index_covers_leaves_only_with_contiguous_indices() -> None:
    index = Taxonomy.model_validate(_valid()).label_index()
    assert "MACHINE" not in index
    assert sorted(index.values()) == list(range(len(index)))
    assert list(index) == sorted(index)


def test_label_index_and_fingerprint_ignore_file_order() -> None:
    forward = Taxonomy.model_validate(_valid())
    reversed_data = _valid()
    reversed_data["causes"].reverse()
    backward = Taxonomy.model_validate(reversed_data)
    assert forward.label_index() == backward.label_index()
    assert forward.fingerprint == backward.fingerprint


def test_fingerprint_is_short_hex_and_tracks_leaf_set_only() -> None:
    base = Taxonomy.model_validate(_valid())
    assert re.fullmatch(r"[0-9a-f]{16}", base.fingerprint)

    renamed = _valid()
    renamed["version"] = "test-2"
    renamed["causes"][1]["name_en"] = "Different display name"
    assert Taxonomy.model_validate(renamed).fingerprint == base.fingerprint

    extended = Taxonomy.model_validate(_with(_cause("MACHINE.SPINDLE_LOAD", "machine", "MACHINE")))
    assert extended.fingerprint != base.fingerprint


# --- structural validation ---------------------------------------------------------


def test_duplicate_codes_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate cause codes"):
        Taxonomy.model_validate(_with(_cause("MACHINE.TOOL_WEAR", "machine", "MACHINE")))


def test_unknown_parent_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown parent codes"):
        Taxonomy.model_validate(_with(_cause("METHOD.WRONG_SETUP", "method", "METHOD")))


@pytest.mark.parametrize(
    "bad_code", ["machine.tool_wear", "MACHINE..X", "MACHINE.", "1MACHINE", "MACHINE-X"]
)
def test_malformed_codes_rejected(bad_code: str) -> None:
    with pytest.raises(ValidationError):
        Taxonomy.model_validate(_with(_cause(bad_code, "machine")))


def test_first_segment_must_equal_category() -> None:
    with pytest.raises(ValidationError, match="first segment must be 'MACHINE'"):
        Taxonomy.model_validate(_with(_cause("MATERIAL.WRONG_PLACE", "machine")))


def test_child_code_must_start_with_parent_and_dot() -> None:
    with pytest.raises(ValidationError, match="must start with parent"):
        Taxonomy.model_validate(
            _with(
                _cause("MACHINE.SPINDLE", "machine", "MACHINE"),
                _cause("MACHINE.COOLANT_DRIFT", "machine", "MACHINE.SPINDLE"),
            )
        )


def test_prefix_without_dot_is_not_a_valid_child() -> None:
    with pytest.raises(ValidationError, match="must start with parent"):
        Taxonomy.model_validate(
            _with(
                _cause("MACHINE.TOOL_WEAR.SEVERE", "machine", "MACHINE.TOOL_WEAR"),
                _cause("MACHINE.TOOL_WEARX", "machine", "MACHINE.TOOL_WEAR"),
            )
        )


def test_child_category_must_equal_parent_category() -> None:
    # A different category always also breaks the first-segment rule, which fires
    # first; the explicit category check is kept as a second guard.
    with pytest.raises(ValidationError, match="first segment|category must equal"):
        Taxonomy.model_validate(_with(_cause("MACHINE.OPERATOR_ERROR", "personnel", "MACHINE")))


def test_orphan_next_to_existing_prefix_rejected() -> None:
    # (a) MACHINE.TOOL_WEAR exists, so MACHINE.TOOL_WEAR.SEVERE may not omit its parent.
    with pytest.raises(ValidationError, match="parent must be 'MACHINE.TOOL_WEAR'"):
        Taxonomy.model_validate(_with(_cause("MACHINE.TOOL_WEAR.SEVERE", "machine")))


def test_child_with_longest_existing_prefix_as_parent_loads() -> None:
    # (b) The same cause with the correct parent is valid and becomes a leaf.
    taxonomy = Taxonomy.model_validate(
        _with(_cause("MACHINE.TOOL_WEAR.SEVERE", "machine", "MACHINE.TOOL_WEAR"))
    )
    assert taxonomy.get("MACHINE.TOOL_WEAR.SEVERE").parent == "MACHINE.TOOL_WEAR"
    assert "MACHINE.TOOL_WEAR" not in taxonomy.leaf_codes
    assert "MACHINE.TOOL_WEAR.SEVERE" in taxonomy.leaf_codes


def test_code_without_existing_prefixes_may_have_no_parent() -> None:
    # (c) Neither MATERIAL nor MATERIAL.SUPPLIER exists as a cause, so no parent is needed.
    taxonomy = Taxonomy.model_validate(_with(_cause("MATERIAL.SUPPLIER.LATE_CERT", "material")))
    assert taxonomy.get("MATERIAL.SUPPLIER.LATE_CERT").parent is None
    assert taxonomy.get("MATERIAL.BAD_BATCH").parent is None


def test_parent_may_not_skip_an_existing_level() -> None:
    # Extra case: MACHINE is a valid prefix, but MACHINE.TOOL_WEAR is longer and exists.
    with pytest.raises(ValidationError, match="parent must be 'MACHINE.TOOL_WEAR'"):
        Taxonomy.model_validate(_with(_cause("MACHINE.TOOL_WEAR.SEVERE", "machine", "MACHINE")))


def test_cycles_and_self_parents_are_impossible() -> None:
    with pytest.raises(ValidationError, match="must start with parent"):
        Taxonomy.model_validate(_with(_cause("MACHINE.LOOP", "machine", "MACHINE.LOOP")))
    with pytest.raises(ValidationError, match="must start with parent"):
        Taxonomy.model_validate(
            _with(
                _cause("MACHINE.A", "machine", "MACHINE.A.B"),
                _cause("MACHINE.A.B", "machine", "MACHINE.A"),
            )
        )


def test_unknown_fields_and_empty_lists_rejected() -> None:
    extra = _valid()
    extra["causes"][0]["colour"] = "red"
    with pytest.raises(ValidationError):
        Taxonomy.model_validate(extra)
    with pytest.raises(ValidationError):
        Taxonomy.model_validate({"version": "x", "causes": []})


# --- file handling -----------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_taxonomy(tmp_path / "absent.yaml")


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_taxonomy(path)


def test_python_object_tags_are_not_executed(tmp_path: Path) -> None:
    path = tmp_path / "evil.yaml"
    path.write_text("version: !!python/object/apply:os.getcwd []\ncauses: []\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_taxonomy(path)
