"""VisionClient against a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from iqs_dashboard.client import VisionAPIError, VisionClient
from iqs_dashboard.settings import DashboardSettings


def _client(handler: object) -> VisionClient:
    http = httpx.Client(base_url="http://vision", transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return VisionClient(http, "k1")


def test_inspect_sends_multipart_and_key() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["key"] = req.headers["X-API-Key"]
        seen["path"] = req.url.path
        body = req.content
        seen["has_file"] = b'name="file"' in body and b"PNGDATA" in body
        seen["has_cat"] = b'name="category"' in body and b"bottle" in body
        return httpx.Response(200, json={"decision": "normal"})

    out = _client(handler).inspect(b"PNGDATA", "bottle")
    assert out == {"decision": "normal"}
    assert seen == {"key": "k1", "path": "/v1/inspections", "has_file": True, "has_cat": True}


def test_errors_carry_status_and_detail() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(415, json={"detail": "unsupported type"})

    with pytest.raises(VisionAPIError) as err:
        _client(handler).inspect(b"x", "bottle", "x.pdf", "application/pdf")
    assert err.value.status == 415 and err.value.detail == "unsupported type"


def test_non_json_error_body() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(VisionAPIError, match="502"):
        _client(handler).models()


def test_models_and_capabilities() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/models":
            return httpx.Response(200, json=[{"category": "bottle"}])
        return httpx.Response(200, content=json.dumps({"features": {"inspections": True}}))

    c = _client(handler)
    assert c.models() == [{"category": "bottle"}]
    assert c.capabilities()["features"]["inspections"] is True


def test_from_settings_uses_first_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQS_API__KEYS", "new,old")
    monkeypatch.setenv("IQS_DASHBOARD__VISION_URL", "http://vision:8001")
    c = VisionClient.from_settings(DashboardSettings(_env_file=None))
    assert c._headers == {"X-API-Key": "new"}
    assert str(c._http.base_url) == "http://vision:8001"
