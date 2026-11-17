from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iqs_common.settings import CommonSettings, get_settings


def test_defaults() -> None:
    s = CommonSettings()
    assert s.environment == "dev"
    assert s.logging.level == "INFO"
    assert s.logging.json_output is True
    assert s.database.host == "localhost"
    assert s.database.port == 5432
    assert s.neo4j.uri == "bolt://localhost:7687"
    assert s.taxonomy_path == Path("configs/taxonomy/causes.yaml")


def test_nested_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQS_ENVIRONMENT", "test")
    monkeypatch.setenv("IQS_DATABASE__PORT", "6543")
    monkeypatch.setenv("IQS_LOGGING__JSON_OUTPUT", "false")
    s = CommonSettings()
    assert s.environment == "test"
    assert s.database.port == 6543
    assert s.logging.json_output is False


def test_dotenv_file_in_working_directory_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("IQS_DATABASE__NAME=from_dotenv\n", encoding="utf-8")
    assert CommonSettings().database.name == "from_dotenv"


def test_environment_variable_beats_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("IQS_DATABASE__NAME=from_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("IQS_DATABASE__NAME", "from_env")
    assert CommonSettings().database.name == "from_env"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("IQS_DATABASE__PORT", "70000"),
        ("IQS_ENVIRONMENT", "staging"),
        ("IQS_LOGGING__LEVEL", "LOUD"),
    ],
)
def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        CommonSettings()


def test_password_is_hidden_but_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQS_DATABASE__PASSWORD", "s3cr3t-value")
    s = CommonSettings()
    url = s.database.sqlalchemy_url()
    assert "s3cr3t-value" not in repr(s)
    assert "s3cr3t-value" not in str(url)
    assert url.password == "s3cr3t-value"  # pragma: allowlist secret (test value)


def test_url_uses_psycopg3_driver_and_omits_empty_password() -> None:
    url = CommonSettings().database.sqlalchemy_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.password is None
    assert (url.host, url.port, url.database, url.username) == ("localhost", 5432, "iqs", "iqs")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
