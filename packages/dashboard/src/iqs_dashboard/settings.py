"""Dashboard settings (IQS_DASHBOARD__*). The dashboard talks only to the service APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from iqs_common.settings import CommonSettings


class DashboardConfig(BaseModel):
    vision_url: str = "http://localhost:8001"
    timeout_s: float = Field(default=60.0, gt=0)


class DashboardSettings(CommonSettings):
    dashboard: DashboardConfig = DashboardConfig()
