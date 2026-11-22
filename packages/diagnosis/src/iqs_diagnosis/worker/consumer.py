"""``new_defect_cluster`` consumer (step 7): open exactly one case and run the agent.

Order of work for one event:

1. ``open_vision_case`` - the event ID and the case are written in **one transaction**;
   a redelivered event, or a second event for the same cluster, opens no second case.
2. run the agent up to the engineer's confirmation, **only if** the case has not been run
   (``phase == not_started``). If the process dies between 1 and 2, the worker's start-up
   sweep (:func:`resume_unrun_cases`) or the next redelivery runs it; the agent itself is
   never run twice for one case.

Errors raise, so the outbox worker retries the event and parks it as ``failed`` after the
maximum attempts (nothing is dropped silently).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from iqs_common.events import Event, NewDefectClusterEvent
from iqs_common.log import get_logger
from iqs_diagnosis.agent.graph import DiagnosisAgent
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.ingestion.vision import VISION_SOURCE, case_from_cluster_event
from iqs_diagnosis.persistence.repository import CaseRepository, CaseStatus

_log = get_logger("iqs_diagnosis.worker")


@dataclass
class VisionCaseConsumer:
    agent: DiagnosisAgent
    repository: CaseRepository
    runs_root: Path | None
    baseline_hours: float

    def __call__(self, event: Event) -> None:
        if not isinstance(event, NewDefectClusterEvent):
            raise TypeError(f"unexpected event type {event.type}")
        case = case_from_cluster_event(event.payload, self.runs_root, self.baseline_hours)
        created = self.repository.open_vision_case(event.event_id, case)
        _log.info(
            "vision_case_opened" if created else "vision_case_exists",
            case_id=case.case_id,
            cluster_id=case.source_cluster_id,
            event_id=str(event.event_id),
        )
        stored = self.repository.get(case.case_id)
        if stored is not None:
            self._run_once(stored)

    def _run_once(self, case: DiagnosisCase) -> None:
        if self.agent.phase(case.case_id) != "not_started":
            return
        state = asyncio.run(self.agent.run(case))
        _log.info(
            "vision_case_run",
            case_id=case.case_id,
            phase=self.agent.phase(case.case_id),
            novel=state.get("novel"),
            top3=state.get("top3"),
        )

    def resume_unrun_cases(self) -> int:
        """Run vision cases that were registered but never run (crash recovery)."""
        n = 0
        for case, status in self.repository.list_cases(source=VISION_SOURCE):
            if status == CaseStatus.REGISTERED and self.agent.phase(case.case_id) == "not_started":
                self._run_once(case)
                n += 1
        return n
