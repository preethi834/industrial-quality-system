"""``python -m iqs_diagnosis.worker``: deliver ``new_defect_cluster`` events (compose, 7g)."""

from __future__ import annotations

import sys

from iqs_common.db import create_db_engine, create_session_factory
from iqs_common.events import EventType
from iqs_common.log import configure_logging, get_logger
from iqs_common.outbox_worker import SqlOutbox, run_worker
from iqs_diagnosis.api.wiring import build_case_service
from iqs_diagnosis.settings import get_diagnosis_settings
from iqs_diagnosis.worker.consumer import VisionCaseConsumer


def main() -> int:
    settings = get_diagnosis_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    log = get_logger("iqs_diagnosis.worker")
    service = build_case_service(settings)
    if service is None or settings.diagnosis.store != "postgres":
        log.error("worker_needs_library_and_postgres")
        return 2
    consumer = VisionCaseConsumer(
        service.agent,
        service.repository,
        settings.diagnosis.runs_root,
        settings.diagnosis.vision_baseline_hours,
    )
    log.info("worker_started", resumed=consumer.resume_unrun_cases())
    factory = create_session_factory(create_db_engine(settings.database))
    run_worker(SqlOutbox(factory), {EventType.NEW_DEFECT_CLUSTER}, consumer, poll_s=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
