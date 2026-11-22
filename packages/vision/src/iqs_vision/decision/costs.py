"""Cost assumptions for the accept / reject / review decision.

Every number is an assumption, never a measurement: each value carries a note that must say
so, and reports cite the sha256 of the YAML file the numbers came from (step 6 gate 3).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Assumed(BaseModel):
    """A value that is assumed, not measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(ge=0)
    note: str

    @field_validator("note")
    @classmethod
    def _says_assumption(cls, v: str) -> str:
        if not v.strip().lower().startswith("assumption"):
            raise ValueError("note must start with 'assumption'")
        return v


class CostConfig(BaseModel):
    """Costs per part (currency units) and human-review error rates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    currency: str = "EUR"
    cost_false_reject: Assumed  # a good part is scrapped
    cost_escape: Assumed  # a defective part is shipped
    cost_review: Assumed  # one human look
    review_miss_rate: Assumed  # reviewer passes a defect
    review_false_reject_rate: Assumed  # reviewer rejects a good part
    defect_rate: Assumed  # production defect rate pi

    @model_validator(mode="after")
    def _ranges(self) -> CostConfig:
        if self.cost_escape.value <= 0 or self.cost_false_reject.value <= 0:
            raise ValueError("cost_escape and cost_false_reject must be > 0")
        for name in ("review_miss_rate", "review_false_reject_rate"):
            if not 0 <= getattr(self, name).value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 < self.defect_rate.value < 1:
            raise ValueError("defect_rate must be in (0, 1)")
        return self

    # Plain floats for the arithmetic.
    @property
    def c_fr(self) -> float:
        return self.cost_false_reject.value

    @property
    def c_esc(self) -> float:
        return self.cost_escape.value

    @property
    def c_rev(self) -> float:
        return self.cost_review.value

    @property
    def eps_miss(self) -> float:
        return self.review_miss_rate.value

    @property
    def eps_fr(self) -> float:
        return self.review_false_reject_rate.value

    @property
    def pi(self) -> float:
        return self.defect_rate.value

    def with_values(self, **values: float) -> CostConfig:
        """Copy with some values replaced (for sensitivity sweeps); notes are kept."""
        data: dict[str, Any] = self.model_dump()
        for name, v in values.items():
            data[name] = {**data[name], "value": v}
        data["id"] = f"{self.id}+sweep"
        return CostConfig.model_validate(data)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cost_config(path: Path) -> tuple[CostConfig, str]:
    """Load a cost YAML; returns the config and the sha256 of the file bytes."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CostConfig.model_validate(raw), sha256_file(path)
