"""8D drafting: agent sections drafted, engineer sections left empty, completeness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.reporting.eight_d import (
    AGENT_SECTIONS,
    SECTION_KEYS,
    EightDReport,
    Section,
    SectionStatus,
    check_completeness,
    draft_8d,
)
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary

from .conftest import TAXONOMY_PATH, VOCABULARY_PATH

CAUSES = [c.code for c in load_taxonomy(TAXONOMY_PATH).causes]


def _report(reference_runs: dict[str, Path], reference_profiles: dict[str, Any]) -> EightDReport:
    case = load_case(reference_runs["fixture_offset_m2b"])
    assert case is not None
    profile, _ = reference_profiles["fixture_offset_m2b"]
    vocab = load_vocabulary(VOCABULARY_PATH)
    return draft_8d(
        case,
        profile,
        top3=[
            ("MACHINE.FIXTURE_MISALIGNMENT", 0.9),
            ("MACHINE.TOOL_WEAR", 0.05),
            ("MATERIAL.BAD_BATCH", 0.05),
        ],
        novel=False,
        max_similarity=0.93,
        neighbours=[{"case_id": "X", "cause_code": "MACHINE.FIXTURE_MISALIGNMENT", "score": 0.03}],
        cause_names={c.code: c.name_en for c in load_taxonomy(TAXONOMY_PATH).causes},
        symptom_names={s.code: s.name_en for s in vocab.symptoms},
    )


def test_draft_fills_agent_sections_only(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> None:
    r = _report(reference_runs, reference_profiles)
    assert tuple(s.key for s in r.sections) == SECTION_KEYS
    for s in r.sections:
        if s.key in AGENT_SECTIONS:
            assert s.status is SectionStatus.DRAFTED and s.content
        else:
            assert s.status is SectionStatus.PENDING_ENGINEER
            assert s.content == "" and s.data == {}
    assert "M2B" in r.section("D2").content
    assert "not confirmed" in r.section("D4").content
    assert r.provenance == "generated"


def test_completeness_full_and_problems(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> None:
    r = _report(reference_runs, reference_profiles)
    c = check_completeness(r, CAUSES)
    assert c.score == 1.0 and c.problems == () and c.drafted_total == 3
    bad = check_completeness(r, ["MACHINE.TOOL_WEAR"])  # candidates outside this list
    assert bad.score == pytest.approx(2 / 3)
    assert any(p.startswith("D4") for p in bad.problems)


def test_pending_section_must_be_empty() -> None:
    with pytest.raises(ValidationError, match="pending"):
        Section(key="D3", title="t", status=SectionStatus.PENDING_ENGINEER, content="invented")


def test_sections_must_be_complete_and_ordered() -> None:
    pending = [Section(key=k, title=k, status=SectionStatus.PENDING_ENGINEER) for k in SECTION_KEYS]
    with pytest.raises(ValidationError, match="D0..D8"):
        EightDReport(case_id="c", sections=tuple(reversed(pending)))
    with pytest.raises(ValidationError):
        Section(key="D9", title="x", status=SectionStatus.DRAFTED, content="x")
    empty = EightDReport(case_id="c", sections=tuple(pending))
    assert check_completeness(empty, CAUSES).score == 0.0
