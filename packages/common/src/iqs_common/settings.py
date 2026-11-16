"""Typed settings shared by all services.

Values come from environment variables (prefix IQS_, nested with "__") and an
optional .env file in the working directory. Each service extends
CommonSettings with its own sections; service-specific paths (for example the
simulator's ground-truth directory) never belong here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

Environment = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LoggingSettings(BaseModel):
    level: LogLevel = "INFO"
    json_output: bool = True


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "iqs"
    user: str = "iqs"
    password: SecretStr = SecretStr("")
    pool_size: int = Field(default=5, ge=1, le=50)

    def sqlalchemy_url(self) -> URL:
        """Connection URL for SQLAlchemy + psycopg 3. str(url) hides the password."""
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value() or None,
            host=self.host,
            port=self.port,
            database=self.name,
        )


class Neo4jSettings(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("")


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IQS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = "dev"
    logging: LoggingSettings = LoggingSettings()
    database: DatabaseSettings = DatabaseSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    taxonomy_path: Path = Path("configs/taxonomy/causes.yaml")


@lru_cache(maxsize=1)
def get_settings() -> CommonSettings:
    """Process-wide settings, read once."""
    return CommonSettings()
