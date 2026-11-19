"""Typed HTTP client for the vision API (the dashboard never touches databases)."""

from __future__ import annotations

from typing import Any

import httpx

from iqs_dashboard.settings import DashboardSettings


class VisionAPIError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"vision API {status}: {detail}")
        self.status = status
        self.detail = detail


class VisionClient:
    def __init__(self, http: httpx.Client, api_key: str) -> None:
        self._http = http
        self._headers = {"X-API-Key": api_key}

    @classmethod
    def from_settings(cls, settings: DashboardSettings) -> VisionClient:
        keys = settings.api.key_list()
        http = httpx.Client(
            base_url=settings.dashboard.vision_url, timeout=settings.dashboard.timeout_s
        )
        return cls(http, keys[0] if keys else "")

    def _check(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = str(r.json().get("detail", r.text))
            except ValueError:
                detail = r.text
            raise VisionAPIError(r.status_code, detail)
        return r.json()

    def health(self) -> dict[str, Any]:
        return dict(self._check(self._http.get("/health")))

    def capabilities(self) -> dict[str, Any]:
        return dict(self._check(self._http.get("/v1/capabilities", headers=self._headers)))

    def models(self) -> list[dict[str, Any]]:
        return list(self._check(self._http.get("/v1/models", headers=self._headers)))

    def inspect(
        self,
        image: bytes,
        category: str,
        filename: str = "image.png",
        content_type: str = "image/png",
    ) -> dict[str, Any]:
        r = self._http.post(
            "/v1/inspections",
            headers=self._headers,
            files={"file": (filename, image, content_type)},
            data={"category": category},
        )
        return dict(self._check(r))


_client: VisionClient | None = None


def get_client() -> VisionClient:
    """Process-wide client built from the environment (tests replace this function)."""
    global _client
    if _client is None:
        _client = VisionClient.from_settings(DashboardSettings())
    return _client
