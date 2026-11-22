"""Cross-service events, delivered through the Postgres outbox (step 0d / step 7).

new_defect_cluster : vision    -> diagnosis  (open a case)
case_confirmed     : diagnosis -> vision     (cluster becomes a named defect class)

Delivery is at-least-once, so consumers must be idempotent on event_id.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from iqs_common.ids import uuid7
from iqs_common.taxonomy import CauseCode


class EventType(StrEnum):
    NEW_DEFECT_CLUSTER = "new_defect_cluster"
    CASE_CONFIRMED = "case_confirmed"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


SIMULATOR_RUN_PATTERN: Final = r"^run-[0-9a-f]{12}$"


class NewDefectClusterPayload(_Strict):
    """A new cluster of unknown anomalies (vision discovery trigger, step 7).

    The step-7 fields (``line_id`` ... ``score_median``) are optional additions to schema
    version 1: no producer or consumer existed before step 7 (ADR-015). ``linkage`` says
    whether ``part_ids`` / ``simulator_run_id`` come from the synthetic image-to-part
    linker; real process linkage does not exist in this project.
    """

    cluster_id: UUID
    category: str = Field(min_length=1)  # product category, e.g. an MVTec class
    size: int = Field(ge=1)
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    inspection_ids: tuple[UUID, ...] = Field(min_length=1, max_length=1000)
    part_ids: tuple[str, ...] = ()  # synthetic simulator part IDs (labelled as such)
    model_version: str = Field(min_length=1)
    line_id: str | None = Field(default=None, min_length=1, max_length=64)
    simulator_run_id: str | None = Field(default=None, pattern=SIMULATOR_RUN_PATTERN)
    linkage: Literal["synthetic_simulator", "none"] = "none"
    representative_inspection_id: UUID | None = None
    score_max: float | None = None
    score_median: float | None = None

    @model_validator(mode="after")
    def _check(self) -> NewDefectClusterPayload:
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be earlier than first_seen")
        if len(self.inspection_ids) > self.size:
            raise ValueError("more inspection_ids than the cluster size")
        linked = bool(self.part_ids) or self.simulator_run_id is not None
        if linked and self.linkage != "synthetic_simulator":
            raise ValueError("part_ids / simulator_run_id require linkage='synthetic_simulator'")
        if (
            self.representative_inspection_id is not None
            and self.representative_inspection_id not in self.inspection_ids
        ):
            raise ValueError("representative_inspection_id must be one of inspection_ids")
        if (
            self.score_max is not None
            and self.score_median is not None
            and self.score_median > self.score_max
        ):
            raise ValueError("score_median must not exceed score_max")
        return self


def default_defect_class_name(category: str, cause_code: str) -> str:
    """Name a confirmed vision cluster gets when the engineer gives none (step 7).

    Both services use this one function, so the vision class and the knowledge-graph
    ``DefectType`` node always carry the same name.
    """
    return f"{category}:{cause_code.split('.')[-1].lower()}"


class CaseConfirmedPayload(_Strict):
    case_id: UUID
    cause_code: CauseCode  # format checked here; existence checked by the consumer
    taxonomy_version: str = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)
    confirmed_at: AwareDatetime
    source_cluster_id: UUID | None = None  # set for vision-originated cases
    defect_class_name: str | None = None

    @model_validator(mode="after")
    def _check(self) -> CaseConfirmedPayload:
        if self.defect_class_name is not None and self.source_cluster_id is None:
            raise ValueError("defect_class_name requires source_cluster_id")
        return self


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _EventBase(_Strict):
    event_id: UUID = Field(default_factory=uuid7)
    schema_version: Literal[1] = 1
    occurred_at: AwareDatetime = Field(default_factory=_utcnow)


class NewDefectClusterEvent(_EventBase):
    type: Literal[EventType.NEW_DEFECT_CLUSTER] = EventType.NEW_DEFECT_CLUSTER
    producer: Literal["vision"] = "vision"
    payload: NewDefectClusterPayload


class CaseConfirmedEvent(_EventBase):
    type: Literal[EventType.CASE_CONFIRMED] = EventType.CASE_CONFIRMED
    producer: Literal["diagnosis"] = "diagnosis"
    payload: CaseConfirmedPayload


Event = Annotated[NewDefectClusterEvent | CaseConfirmedEvent, Field(discriminator="type")]
_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(raw: str | bytes | dict[str, object]) -> Event:
    """Validate an outbox row (JSON text or dict) into the matching event class."""
    if isinstance(raw, dict):
        return _EVENT_ADAPTER.validate_python(raw)
    return _EVENT_ADAPTER.validate_json(raw)
