"""Confirmation feedback (step 7): what the system learns from an engineer's confirmation.

For a confirmed cause (in the taxonomy):

1. Postgres, one transaction, once per case (:mod:`iqs_diagnosis.feedback.store`):
   Dirichlet count +1 for (context, cause), the case into the calibration **candidate**
   pool, and for vision cases a ``case_confirmed`` outbox event (vision names the cluster).
2. Neo4j (idempotent ``MERGE``): ``CONFIRMED_CAUSE``, and for vision cases the
   ``DefectType`` node with ``INDICATES {count}``. If Neo4j is down, the Postgres part still
   commits; the graph write is retried by :meth:`FeedbackService.flush_graph`.

Count context: ``defect_type:<category>/<name>`` for vision cases (what the defect looked
like), ``source:<source>`` otherwise. The counts are **recorded, not used**: the scorer stays
the retrieval-vote ranker until the Bayesian network of step 8a consumes them
(KNOWN_ISSUES #44). The live case library and the active calibration run are not changed
(owner decision D3).

``cause_code = None`` ("no listed cause fits"): the confirmation is logged, nothing is
counted, no graph edge, no event, no candidate (the calibration set covers in-taxonomy
causes only); the vision cluster stays ``alerted``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from iqs_common.events import CaseConfirmedEvent, CaseConfirmedPayload, default_defect_class_name
from iqs_common.log import get_logger
from iqs_common.taxonomy import Taxonomy
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.feedback.store import FeedbackRecord, FeedbackStore
from iqs_diagnosis.graph.repository import ConfirmationWrite, DefectTypeRef, GraphWriter
from iqs_diagnosis.ingestion.vision import VISION_SOURCE, VISUAL_CLUSTER_SYMPTOM

_log = get_logger("iqs_diagnosis.feedback")


def vision_category(case: DiagnosisCase) -> str | None:
    for s in case.reported_symptoms:
        if s.get("code") == VISUAL_CLUSTER_SYMPTOM and isinstance(s.get("category"), str):
            return str(s["category"])
    return None


@dataclass
class FeedbackService:
    store: FeedbackStore
    taxonomy: Taxonomy
    graph: GraphWriter | None = None  # None: graph writes disabled (reported as such)

    def apply(
        self,
        case: DiagnosisCase,
        cause_code: str | None,
        engineer: str,
        defect_class_name: str | None = None,
        confirmed_at: datetime | None = None,
    ) -> dict[str, Any]:
        confirmed_at = confirmed_at or datetime.now(UTC)
        is_vision = case.source == VISION_SOURCE and case.source_cluster_id is not None
        category = vision_category(case) if is_vision else None
        name: str | None = None
        context: str | None = None
        if cause_code is not None:
            if is_vision and category is not None:
                name = defect_class_name or default_defect_class_name(category, cause_code)
                context = f"defect_type:{category}/{name}"
            else:
                context = f"source:{case.source}"
        event = None
        if is_vision and cause_code is not None:
            assert case.source_cluster_id is not None
            event = CaseConfirmedEvent(
                payload=CaseConfirmedPayload(
                    case_id=UUID(case.case_id),
                    cause_code=cause_code,
                    taxonomy_version=self.taxonomy.version,
                    confirmed_by=engineer,
                    confirmed_at=confirmed_at,
                    source_cluster_id=UUID(case.source_cluster_id),
                    defect_class_name=name,
                )
            )
        record = FeedbackRecord(
            case_id=case.case_id,
            source=case.source,
            cause_code=cause_code,
            engineer=engineer,
            confirmed_at=confirmed_at,
            count_context=context,
            category=category,
            defect_class_name=name,
            event_id=None if event is None else event.event_id,
            candidate_added=cause_code is not None,
        )
        stored, first = self.store.apply(record, event)
        graph = self._write_graph(stored)
        _log.info(
            "feedback_applied" if first else "feedback_already_applied",
            case_id=case.case_id,
            cause_code=cause_code,
            context=context,
            count_after=stored.count_after,
            graph=graph["status"],
        )
        out = self.store.get(case.case_id) or stored
        return {**out.to_json(), "first_application": first, "graph": graph}

    def _write_graph(self, rec: FeedbackRecord) -> dict[str, Any]:
        if rec.cause_code is None:
            return {"status": "skipped", "reason": "no cause confirmed"}
        if rec.graph_written_at is not None:
            return {"status": "written", "at": rec.graph_written_at.isoformat()}
        if self.graph is None:
            return {"status": "disabled", "reason": "no graph writer configured"}
        cause_name = next(
            (c.name_en for c in self.taxonomy.causes if c.code == rec.cause_code), rec.cause_code
        )
        defect = (
            DefectTypeRef(rec.category, rec.defect_class_name)
            if rec.category is not None and rec.defect_class_name is not None
            else None
        )
        try:
            self.graph.write_confirmation(
                ConfirmationWrite(
                    case_id=rec.case_id,
                    source=rec.source,
                    cause_code=rec.cause_code,
                    cause_name=cause_name,
                    engineer=rec.engineer,
                    confirmed_at=rec.confirmed_at,
                    defect_type=defect,
                    indicates_count=rec.count_after,
                )
            )
        except Exception as exc:  # retried by flush_graph; Postgres feedback is committed
            _log.warning("graph_write_failed", case_id=rec.case_id, error=type(exc).__name__)
            return {"status": "pending", "error": type(exc).__name__}
        at = datetime.now(UTC)
        self.store.mark_graph_written(rec.case_id, at)
        return {"status": "written", "at": at.isoformat()}

    def flush_graph(self) -> int:
        """Retry graph writes that failed earlier; returns how many succeeded."""
        done = 0
        for rec in self.store.pending_graph():
            if self._write_graph(replace(rec))["status"] == "written":
                done += 1
        return done
