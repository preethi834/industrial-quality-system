"""The shared root-cause taxonomy (blueprint design decision 3).

Every cause the agent, the Bayesian network, conformal prediction and the
simulator can name is one entry here. Free text explains a cause; it is never
the label. The data lives in configs/taxonomy/causes.yaml (step 0c).

Structure rules (enforced on load):
- codes are dotted UPPER_SNAKE segments, e.g. MACHINE.TOOL_WEAR
- the first segment equals the cause's 6M category in upper case
- a child's code starts with "<parent code>." and shares the parent's category
  (this also makes cycles in the hierarchy impossible)
- if any proper prefix of a code exists as a cause, the parent must be the
  longest such prefix (no skipped levels, no orphans next to their parent)
- only leaf causes are labels; parents are groupings
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CAUSE_CODE_PATTERN = r"^[A-Z][A-Z0-9_]*(\.[A-Z][A-Z0-9_]*)*$"
CauseCode = Annotated[str, StringConstraints(pattern=CAUSE_CODE_PATTERN, max_length=64)]


class CauseCategory(StrEnum):
    """6M (Ishikawa) categories used in 8D root-cause analysis."""

    MACHINE = "machine"
    MATERIAL = "material"
    METHOD = "method"
    MEASUREMENT = "measurement"
    ENVIRONMENT = "environment"
    PERSONNEL = "personnel"


class Cause(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: CauseCode
    name_en: str = Field(min_length=1)
    name_de: str = Field(min_length=1)
    category: CauseCategory
    parent: CauseCode | None = None
    description: str = ""
    test_template: str | None = None  # scientist-agent test (step 5)
    simulator_fault: str | None = None  # injectable fault type (step 5)


class Taxonomy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    causes: tuple[Cause, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_structure(self) -> Taxonomy:
        codes = [c.code for c in self.causes]
        duplicates = sorted({c for c in codes if codes.count(c) > 1})
        if duplicates:
            raise ValueError(f"duplicate cause codes: {duplicates}")
        by_code = {c.code: c for c in self.causes}
        unknown_parents = sorted(
            {c.parent for c in self.causes if c.parent is not None and c.parent not in by_code}
        )
        if unknown_parents:
            raise ValueError(f"unknown parent codes: {unknown_parents}")
        for cause in self.causes:
            expected_head = cause.category.value.upper()
            if cause.code.split(".", 1)[0] != expected_head:
                raise ValueError(
                    f"{cause.code}: first segment must be {expected_head!r} "
                    f"for category {cause.category.value!r}"
                )
            if cause.parent is not None:
                if not cause.code.startswith(cause.parent + "."):
                    raise ValueError(
                        f"{cause.code}: code must start with parent {cause.parent!r} + '.'"
                    )
                if by_code[cause.parent].category != cause.category:
                    raise ValueError(f"{cause.code}: category must equal its parent's category")
            segments = cause.code.split(".")
            existing_prefixes = [
                prefix
                for prefix in (".".join(segments[:i]) for i in range(1, len(segments)))
                if prefix in by_code
            ]
            if existing_prefixes and cause.parent != existing_prefixes[-1]:
                raise ValueError(
                    f"{cause.code}: parent must be {existing_prefixes[-1]!r}, the longest "
                    f"existing prefix (got {cause.parent!r})"
                )
        return self

    @cached_property
    def _by_code(self) -> dict[str, Cause]:
        return {c.code: c for c in self.causes}

    def __contains__(self, code: object) -> bool:
        return code in self._by_code

    def get(self, code: str) -> Cause:
        try:
            return self._by_code[code]
        except KeyError:
            raise KeyError(f"unknown cause code: {code!r}") from None

    @property
    def codes(self) -> tuple[str, ...]:
        """All codes (parents and leaves), sorted."""
        return tuple(sorted(self._by_code))

    @cached_property
    def leaf_codes(self) -> tuple[str, ...]:
        """Codes of causes without children, sorted: the label space for models."""
        parents = {c.parent for c in self.causes if c.parent is not None}
        return tuple(sorted(code for code in self._by_code if code not in parents))

    def label_index(self) -> dict[str, int]:
        """Stable label -> index mapping over leaf causes only (decision D5)."""
        return {code: i for i, code in enumerate(self.leaf_codes)}

    @property
    def fingerprint(self) -> str:
        """Short SHA-256 of the sorted leaf codes; stored with calibration records."""
        digest = hashlib.sha256("\n".join(self.leaf_codes).encode("utf-8")).hexdigest()
        return digest[:16]

    def children(self, code: str) -> tuple[Cause, ...]:
        self.get(code)  # raises for unknown codes
        return tuple(c for c in self.causes if c.parent == code)


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Load and validate a taxonomy YAML file (safe_load: no arbitrary Python objects)."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return Taxonomy.model_validate(data)
