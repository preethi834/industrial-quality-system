"""HTTP client for the quality-loop page (step 7): vision clusters + diagnosis cases.

Separate from ``client.py`` (the vision inspection client) so the two can change
independently. The dashboard talks only to the service APIs.
"""

from __future__ import annotations

from typing import Any

import httpx

from iqs_dashboard.settings import DashboardSettings


class LoopAPIError(RuntimeError):
    def __init__(self, service: str, status: int, detail: str) -> None:
        super().__init__(f"{service} API {status}: {detail}")
        self.service = service
        self.status = status
        self.detail = detail


def _check(service: str, r: httpx.Response) -> Any:
    if r.status_code >= 400:
        try:
            detail = str(r.json().get("detail", r.text))
        except ValueError:
            detail = r.text
        raise LoopAPIError(service, r.status_code, detail)
    return r.json()


class QualityLoopClient:
    def __init__(self, vision: httpx.Client, diagnosis: httpx.Client, api_key: str) -> None:
        self._v = vision
        self._d = diagnosis
        self._h = {"X-API-Key": api_key}

    @classmethod
    def from_settings(cls, settings: DashboardSettings) -> QualityLoopClient:
        keys = settings.api.key_list()
        d = settings.dashboard
        return cls(
            httpx.Client(base_url=d.vision_url, timeout=d.timeout_s),
            httpx.Client(base_url=d.diagnosis_url, timeout=d.timeout_s),
            keys[0] if keys else "",
        )

    def vision_capabilities(self) -> dict[str, Any]:
        return dict(_check("vision", self._v.get("/v1/capabilities", headers=self._h)))

    def diagnosis_capabilities(self) -> dict[str, Any]:
        return dict(_check("diagnosis", self._d.get("/v1/capabilities", headers=self._h)))

    def clusters(self) -> list[dict[str, Any]]:
        return list(_check("vision", self._v.get("/v1/clusters", headers=self._h)))

    def vision_cases(self) -> list[dict[str, Any]]:
        r = self._d.get("/v1/cases", headers=self._h, params={"source": "vision"})
        return list(_check("diagnosis", r))

    def cause_set(self, case_id: str) -> dict[str, Any]:
        return dict(
            _check("diagnosis", self._d.get(f"/v1/cases/{case_id}/cause-set", headers=self._h))
        )

    def confirm(
        self, case_id: str, cause_code: str | None, engineer: str, defect_class_name: str | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"cause_code": cause_code, "engineer": engineer}
        if defect_class_name:
            body["defect_class_name"] = defect_class_name
        r = self._d.post(f"/v1/cases/{case_id}/confirm", headers=self._h, json=body)
        return dict(_check("diagnosis", r))


_client: QualityLoopClient | None = None


def get_loop_client() -> QualityLoopClient:
    """Process-wide client built from the environment (tests replace this function)."""
    global _client
    if _client is None:
        _client = QualityLoopClient.from_settings(DashboardSettings())
    return _client
