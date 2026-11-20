"""Symptom vocabulary (``configs/diagnosis/symptoms.yaml``): codes and the rules behind them."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from iqs_diagnosis.domain.case import SymptomObservation
from iqs_diagnosis.symptoms.simulator_profile import FEATURE_NAMES

SymptomCode = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
Op = Literal[">=", ">", "<=", "<"]
_OPS: Final[dict[str, Callable[[float, float], bool]]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}


class Symptom(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: SymptomCode
    name_en: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source: Literal["case", "features"] = "features"
    all: tuple[tuple[str, Op, float], ...] = ()
    where: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Symptom:
        if self.source == "case" and self.all:
            raise ValueError(f"{self.code}: a case symptom has no feature rule")
        if self.source == "features" and not self.all:
            raise ValueError(f"{self.code}: a feature symptom needs at least one condition")
        unknown = sorted({f for f, _, _ in self.all} - set(FEATURE_NAMES))
        if self.where is not None:
            unknown += [] if self.where in FEATURE_NAMES else [self.where]
        if unknown:
            raise ValueError(f"{self.code}: unknown features {unknown}")
        return self

    def holds(self, features: Mapping[str, float]) -> bool:
        return all(_OPS[op](features[name], value) for name, op, value in self.all)


class Vocabulary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    symptoms: tuple[Symptom, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Vocabulary:
        codes = [s.code for s in self.symptoms]
        dup = sorted({c for c in codes if codes.count(c) > 1})
        if dup:
            raise ValueError(f"duplicate symptom codes: {dup}")
        return self

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(s.code for s in self.symptoms)

    def observe(
        self,
        features: Mapping[str, float],
        where: Mapping[str, str],
        case_codes: set[str],
    ) -> tuple[SymptomObservation, ...]:
        """Symptoms that hold, in vocabulary order, each with its numbers and location."""
        out: list[SymptomObservation] = []
        for s in self.symptoms:
            if s.source == "case":
                if s.code in case_codes:
                    out.append(SymptomObservation(code=s.code))
                continue
            if not s.holds(features):
                continue
            evidence: dict[str, object] = {
                name: round(float(features[name]), 4) for name, _, _ in s.all
            }
            if s.where is not None and s.where in where:
                evidence["where"] = where[s.where]
            out.append(SymptomObservation(code=s.code, evidence=evidence))
        return tuple(out)


def load_vocabulary(path: Path) -> Vocabulary:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return Vocabulary.model_validate(data)
