"""8D report (D0-D8): schema, template drafting and completeness check.

Step 3c drafts deterministically from the case, the symptom profile and the cause
ranking; no LLM is involved (LLM drafting via Ollama follows in 3e). The agent only
drafts what it has evidence for:

- D0 (symptom / alarm), D2 (problem description) and D4 (root-cause candidates) are
  drafted;
- D1, D3, D5-D8 need people and decisions (team, containment, actions, closure) and are
  marked ``pending_engineer`` with empty content. They are never filled with invented
  text.

D4 lists *candidate* causes. A cause becomes confirmed only through the engineer's
confirmation (and, from step 5b, the scientist agent's data tests).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iqs_diagnosis.domain.case import DiagnosisCase, SymptomProfile

SECTION_KEYS: Final = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
SECTION_TITLES: Final[dict[str, str]] = {
    "D0": "Symptom and emergency response",
    "D1": "Team",
    "D2": "Problem description",
    "D3": "Containment actions",
    "D4": "Root cause (candidates)",
    "D5": "Permanent corrective actions",
    "D6": "Implementation and validation",
    "D7": "Prevent recurrence",
    "D8": "Closure and recognition",
}
# Sections the agent is responsible for drafting; the rest need an engineer.
AGENT_SECTIONS: Final = ("D0", "D2", "D4")
TEMPLATE_VERSION: Final = "template-v1"


class SectionStatus(StrEnum):
    DRAFTED = "drafted"
    PENDING_ENGINEER = "pending_engineer"


class Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    title: str
    status: SectionStatus
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Section:
        if self.key not in SECTION_KEYS:
            raise ValueError(f"unknown 8D section {self.key!r}")
        if self.status is SectionStatus.PENDING_ENGINEER and (self.content or self.data):
            raise ValueError(f"{self.key}: a pending section must be empty")
        return self


class EightDReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    version: int = Field(default=1, ge=1)
    language: str = "en"
    generated_by: str = TEMPLATE_VERSION
    provenance: str = "generated"  # never presented as an engineer-written report
    sections: tuple[Section, ...]

    @model_validator(mode="after")
    def _order(self) -> EightDReport:
        if tuple(s.key for s in self.sections) != SECTION_KEYS:
            raise ValueError("sections must be exactly D0..D8 in order")
        return self

    def section(self, key: str) -> Section:
        return self.sections[SECTION_KEYS.index(key)]


class Completeness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_sections_ok: int
    agent_sections_total: int
    drafted_total: int
    problems: tuple[str, ...]

    @property
    def score(self) -> float:
        return self.agent_sections_ok / self.agent_sections_total


def check_completeness(report: EightDReport, cause_codes: Sequence[str]) -> Completeness:
    """Agent-section completeness: D0/D2/D4 drafted with the content they must carry."""
    problems: list[str] = []
    ok = 0
    for key in AGENT_SECTIONS:
        s = report.section(key)
        missing: list[str] = []
        if s.status is not SectionStatus.DRAFTED or not s.content.strip():
            missing.append("not drafted")
        elif key == "D0" and "alarm_ts" not in s.data:
            missing.append("no alarm time")
        elif key == "D2" and not s.data.get("symptoms"):
            missing.append("no symptoms")
        elif key == "D4":
            cands = s.data.get("candidates") or []
            if not cands or any(c.get("cause_code") not in cause_codes for c in cands):
                missing.append("candidates missing or outside the taxonomy")
            if "novel" not in s.data:
                missing.append("no novelty statement")
        if missing:
            problems.extend(f"{key}: {m}" for m in missing)
        else:
            ok += 1
    drafted = sum(s.status is SectionStatus.DRAFTED for s in report.sections)
    return Completeness(
        agent_sections_ok=ok,
        agent_sections_total=len(AGENT_SECTIONS),
        drafted_total=drafted,
        problems=tuple(problems),
    )


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def draft_8d(
    case: DiagnosisCase,
    profile: SymptomProfile,
    *,
    top3: Sequence[tuple[str, float]],
    novel: bool,
    max_similarity: float,
    neighbours: Sequence[Mapping[str, Any]],
    cause_names: Mapping[str, str],
    symptom_names: Mapping[str, str],
    cause_set: Mapping[str, Any] | None = None,
    scientist: Mapping[str, Any] | None = None,
) -> EightDReport:
    rate = next((s for s in case.reported_symptoms if s.get("code") == "reject_rate_increase"), {})
    alarm = case.alarm_ts.isoformat() if case.alarm_ts else None
    d0_text = f"{case.title}. Reject-rate alarm at {alarm}."
    if rate:
        d0_text += (
            f" Reject rate {_pct(float(rate['rate_before']))} before the alarm window, "
            f"{_pct(float(rate['rate_after']))} since {rate.get('window_start')}."
        )
    d0 = Section(
        key="D0",
        title=SECTION_TITLES["D0"],
        status=SectionStatus.DRAFTED,
        content=d0_text,
        data={"alarm_ts": alarm, "line_id": case.line_id, **({"rates": rate} if rate else {})},
    )

    lines = []
    symptoms = []
    for s in profile.symptoms:
        where = s.evidence.get("where")
        name = symptom_names.get(s.code, s.code)
        lines.append(f"- {name}" + (f" (at {where})" if where else ""))
        symptoms.append({"code": s.code, **({"where": where} if where else {})})
    d2 = Section(
        key="D2",
        title=SECTION_TITLES["D2"],
        status=SectionStatus.DRAFTED,
        content="Observed in the process data (reference period vs alarm window):\n"
        + "\n".join(lines),
        data={"symptoms": symptoms, "window_start": str(case.window_start)},
    )

    cand_lines = [
        f"{i + 1}. {cause_names.get(code, code)} ({code}), vote share {p:.2f}"
        for i, (code, p) in enumerate(top3)
    ]
    novelty_line = (
        "No close past case was found (max similarity "
        f"{max_similarity:.2f}): this may be a new problem; the candidates are weak."
        if novel
        else f"Closest past case similarity {max_similarity:.2f}."
    )
    set_line = ""
    if cause_set is not None:
        if cause_set["causes"]:
            names = ", ".join(f"{cause_names.get(c, c)} ({c})" for c in cause_set["causes"])
            set_line = (
                f"\nCause set ({cause_set['method'].upper()}, "
                f"{100 * (1 - cause_set['alpha']):.0f} % target): {names}."
            )
        else:
            set_line = (
                "\nCause set is empty: no cause reaches the calibrated threshold. "
                "Treat as a possible new problem and escalate."
            )
        if cause_set.get("lac_empty") and cause_set["causes"]:
            set_line += (
                "\nUncertain / possible new problem: no single cause reaches the LAC threshold."
            )
    sci_line = ""
    if scientist is not None and scientist.get("hypotheses"):
        parts = []
        for h in scientist["hypotheses"]:
            res = h["result"]
            where = f" at {res['target']}" if res.get("target") else ""
            p = "" if h["p_adjusted"] is None else f", adjusted p = {h['p_adjusted']:.2g}"
            parts.append(
                f"- {cause_names.get(h['cause_code'], h['cause_code'])}: {h['verdict']}"
                f" ({res['test']}{where}{p})"
            )
        best = scientist.get("data_supported_cause")
        sci_line = (
            f"\nData check (Benjamini-Hochberg, q = {scientist['q']}):\n"
            + "\n".join(parts)
            + (
                f"\nData-supported cause: {cause_names.get(best, best)} ({best})."
                if best
                else "\nNo candidate is supported by the data check."
            )
        )
    d4 = Section(
        key="D4",
        title=SECTION_TITLES["D4"],
        status=SectionStatus.DRAFTED,
        content="Candidate causes from similar confirmed past cases (not confirmed):\n"
        + "\n".join(cand_lines)
        + "\n"
        + novelty_line
        + set_line
        + sci_line,
        data={
            "candidates": [{"cause_code": c, "probability": round(p, 4)} for c, p in top3],
            "novel": novel,
            "max_similarity": round(max_similarity, 4),
            "neighbours": [dict(n) for n in neighbours],
            **({"cause_set": dict(cause_set)} if cause_set is not None else {}),
            **(
                {
                    "data_check": [
                        {
                            "cause_code": h["cause_code"],
                            "verdict": h["verdict"],
                            "p_adjusted": h["p_adjusted"],
                            "test": h["result"]["test"],
                            "target": h["result"].get("target"),
                        }
                        for h in scientist["hypotheses"]
                    ],
                    "data_supported_cause": scientist.get("data_supported_cause"),
                }
                if scientist is not None
                else {}
            ),
        },
    )
    drafted = {"D0": d0, "D2": d2, "D4": d4}
    sections = tuple(
        drafted.get(k)
        or Section(key=k, title=SECTION_TITLES[k], status=SectionStatus.PENDING_ENGINEER)
        for k in SECTION_KEYS
    )
    return EightDReport(case_id=case.case_id, sections=sections)
