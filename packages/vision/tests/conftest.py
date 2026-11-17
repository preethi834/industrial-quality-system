"""Test isolation: no IQS_* variables, no .env, logging restored after each test."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from iqs_common.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("IQS_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    get_settings.cache_clear()
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    structlog.reset_defaults()
