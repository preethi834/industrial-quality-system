"""Scientist evaluation metrics on hand-made cases."""

from __future__ import annotations

from iqs_diagnosis.evaluation.scientist_eval import ScientistCase, summarize, templates_alone_pick


def _case(
    truth: str, verdicts: dict[str, str], p: dict[str, float | None], pick: str | None
) -> ScientistCase:
    return ScientistCase(
        case_id=f"c-{truth}-{pick}",
        fault="f",
        truth=truth,
        known=True,
        agent_top3=["B", "A", "C"],
        all_verdicts=verdicts,
        all_p_adjusted=p,
        agent_verdicts={},
        scientist_pick=pick,
    )


def test_summary_counts() -> None:
    c1 = _case(
        "A",
        {"A": "supported", "B": "rejected", "C": "inconclusive"},
        {"A": 0.001, "B": 0.5, "C": None},
        "A",
    )
    c2 = _case(
        "A",
        {"A": "rejected", "B": "supported", "C": "rejected"},
        {"A": 0.3, "B": 0.01, "C": 0.9},
        "B",
    )
    s = summarize([c1, c2])["by_fault"]["ALL"]
    assert s["correct_cause_supported"]["k"] == 1
    assert (s["wrong_causes_rejected"]["k"], s["wrong_causes_rejected"]["n"]) == (2, 4)
    assert s["wrong_causes_supported"]["k"] == 1
    assert s["agent_top1_accuracy"]["k"] == 0  # agent top-1 is B
    assert s["combined_accuracy"]["k"] == 1  # c1 picks A
    assert templates_alone_pick(c2) == "B" and templates_alone_pick(c1) == "A"
