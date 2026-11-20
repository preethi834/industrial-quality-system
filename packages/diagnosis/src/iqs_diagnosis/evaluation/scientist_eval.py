"""Scientist evaluation metrics (step 5 recorded metrics).

Per case we have the true (primary) cause and scientist reports in two modes:

- **all causes**: every taxonomy cause tested, BH over 10;
- **agent**: the agent's top-3 causes tested, BH over 3.

Recorded:
- share of correct causes *supported* and of wrong causes *rejected* (and supported /
  inconclusive), per fault type, in "all causes" mode;
- root-cause accuracy per fault type: agent top-1; scientist pick (highest-ranked
  supported of the top 3) when it exists; combined answer (scientist pick, else agent
  top-1); templates alone (supported cause with the smallest adjusted p).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from iqs_diagnosis.evaluation.metrics import Proportion, wilson


class ScientistCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    fault: str
    truth: str
    known: bool  # truth among the library's causes
    agent_top3: list[str]
    all_verdicts: dict[str, str]  # cause -> verdict (all-causes mode)
    all_p_adjusted: dict[str, float | None]
    agent_verdicts: dict[str, str]  # cause -> verdict (agent mode, top 3)
    scientist_pick: str | None


def templates_alone_pick(c: ScientistCase) -> str | None:
    sup = [k for k, v in c.all_verdicts.items() if v == "supported"]
    if not sup:
        return None
    return min(sup, key=lambda k: (c.all_p_adjusted[k] or 1.0, k))


def summarize(cases: Sequence[ScientistCase]) -> dict[str, Any]:
    def prop(k: int, n: int) -> dict[str, Any]:
        p: Proportion = wilson(k, n)
        return p.model_dump(mode="json")

    faults = sorted({c.fault for c in cases})
    by_fault: dict[str, Any] = {}
    for f in [*faults, "ALL"]:
        cs = [c for c in cases if f in ("ALL", c.fault)]
        n = len(cs)
        correct_sup = sum(c.all_verdicts[c.truth] == "supported" for c in cs)
        correct_inc = sum(c.all_verdicts[c.truth] == "inconclusive" for c in cs)
        wrong = [(c, k) for c in cs for k in c.all_verdicts if k != c.truth]
        w_rej = sum(c.all_verdicts[k] == "rejected" for c, k in wrong)
        w_sup = sum(c.all_verdicts[k] == "supported" for c, k in wrong)
        w_inc = sum(c.all_verdicts[k] == "inconclusive" for c, k in wrong)
        picks = [c for c in cs if c.scientist_pick is not None]
        combined = sum(
            (c.scientist_pick if c.scientist_pick else c.agent_top3[0]) == c.truth for c in cs
        )
        alone = [templates_alone_pick(c) for c in cs]
        by_fault[f] = {
            "cases": n,
            "correct_cause_supported": prop(correct_sup, n),
            "correct_cause_inconclusive": prop(correct_inc, n),
            "wrong_causes_rejected": prop(w_rej, len(wrong)),
            "wrong_causes_supported": prop(w_sup, len(wrong)),
            "wrong_causes_inconclusive": prop(w_inc, len(wrong)),
            "agent_top1_accuracy": prop(sum(c.agent_top3[0] == c.truth for c in cs), n),
            "scientist_pick_rate": prop(len(picks), n),
            "scientist_pick_accuracy": prop(
                sum(c.scientist_pick == c.truth for c in picks), len(picks)
            ),
            "combined_accuracy": prop(combined, n),
            "templates_alone_accuracy": prop(
                sum(a == c.truth for a, c in zip(alone, cs, strict=True)), n
            ),
            "templates_alone_no_pick": prop(sum(a is None for a in alone), n),
        }
    return {"by_fault": by_fault}
