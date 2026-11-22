"""Contract test for the cross-service events (step 7): run by this service's suite."""

from __future__ import annotations

from pathlib import Path

from iqs_common.contracts import check_contracts, contracts_dir


def test_event_contracts_hold() -> None:
    assert check_contracts(contracts_dir(Path(__file__).resolve())) == []
