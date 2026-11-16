from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from iqs_common.log import bind_context, clear_context, configure_logging, get_logger


def _json_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_json_line_has_standard_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    get_logger("iqs.test").info("hello", part_id="P-1")
    (record,) = _json_lines(capsys)
    assert record["event"] == "hello"
    assert record["level"] == "info"
    assert record["logger"] == "iqs.test"
    assert record["part_id"] == "P-1"
    assert record["timestamp"].endswith("Z")


def test_bound_context_appears_and_clears(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    log = get_logger("iqs.test")
    bind_context(case_id="case-123")
    log.info("with context")
    clear_context()
    log.info("without context")
    first, second = _json_lines(capsys)
    assert first["case_id"] == "case-123"
    assert "case_id" not in second


def test_secrets_are_masked_but_token_counts_are_not(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    get_logger().info(
        "call", password="p", api_key="k", db_password="d", Authorization="a", input_tokens=42
    )
    (record,) = _json_lines(capsys)
    assert record["password"] == "***"
    assert record["api_key"] == "***"
    assert record["db_password"] == "***"
    assert record["Authorization"] == "***"
    assert record["input_tokens"] == 42


def test_exception_produces_traceback_field(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    log = get_logger("iqs.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("operation failed")
    (record,) = _json_lines(capsys)
    assert record["event"] == "operation failed"
    assert record["level"] == "error"
    assert "Traceback (most recent call last)" in record["exception"]
    assert "ValueError: boom" in record["exception"]


def test_stdlib_loggers_are_rendered_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    logging.getLogger("sqlalchemy.engine").warning("pool %s exhausted", "main")
    (record,) = _json_lines(capsys)
    assert record["event"] == "pool main exhausted"
    assert record["logger"] == "sqlalchemy.engine"
    assert record["level"] == "warning"


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json_output=True)
    log = get_logger()
    log.info("hidden")
    log.warning("shown")
    records = _json_lines(capsys)
    assert [r["event"] for r in records] == ["shown"]


def test_console_mode_is_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=False)
    get_logger().info("readable line")
    out = capsys.readouterr().out
    assert "readable line" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.splitlines()[0])
