"""Vision outbox worker: ``python -m iqs_vision.discovery.worker`` (compose, step 7g)."""

from __future__ import annotations

from iqs_common.db import create_db_engine, create_session_factory
from iqs_common.events import EventType
from iqs_common.log import configure_logging, get_logger
from iqs_common.outbox_worker import SqlOutbox, run_worker
from iqs_vision.discovery.consumer import make_case_confirmed_handler
from iqs_vision.discovery.store import SqlDiscoveryStore
from iqs_vision.settings import get_vision_settings


def main() -> None:
    settings = get_vision_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    factory = create_session_factory(create_db_engine(settings.database))
    handler = make_case_confirmed_handler(SqlDiscoveryStore(factory))
    get_logger("iqs_vision.discovery.worker").info("worker_started")
    run_worker(
        SqlOutbox(factory),
        {EventType.CASE_CONFIRMED},
        handler,
        poll_s=settings.vision.discovery.poll_s,
    )


if __name__ == "__main__":
    main()
