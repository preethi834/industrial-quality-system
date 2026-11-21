"""HTTP service plumbing shared by the vision and diagnosis APIs.

Needs the optional extra: iqs-common[api].

- GET /health            liveness: the process answers. No dependencies, no auth.
- GET /ready             readiness: every dependency check passes within the timeout
                         (200), otherwise 503 with the status of each check. No auth.
- GET /v1/capabilities   service, version and which features are implemented. API key.
- Every /v1 route requires the X-API-Key header (keys from IQS_API__KEYS). With no
  keys configured, /v1 rejects every request (fail closed).
- Every response carries X-Request-ID (taken from the request or generated), and the
  ID is bound to all log lines written while handling the request.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from iqs_common.capabilities import feature_status
from iqs_common.ids import uuid7
from iqs_common.log import bind_context, clear_context, get_logger
from iqs_common.settings import ApiSettings

API_VERSION = "v1"
API_KEY_HEADER = "X-API-Key"  # pragma: allowlist secret (header name)
REQUEST_ID_HEADER = "X-Request-ID"
_MAX_REQUEST_ID_LEN = 128

ReadinessCheck = Callable[[], None]
"""Synchronous probe that raises on failure (run in a worker thread)."""

_log = get_logger("iqs_common.api")
# Module level: FastAPI resolves the Annotated dependency below by name.
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]  # name -> "ok" | "error: <ExceptionType>" | "timeout"


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    api_version: str = API_VERSION
    features: dict[str, bool]
    # Step 7 (blueprint §2.9): enabled | disabled | not_installed | missing_artifact,
    # and why a feature is not enabled.
    status: dict[str, str] = {}
    reason: dict[str, str] = {}


def api_key_dependency(settings: ApiSettings) -> Callable[[str | None], None]:
    """FastAPI dependency that accepts any configured key (constant-time compare)."""
    keys = [k.encode() for k in settings.key_list()]

    def require_api_key(supplied: Annotated[str | None, Depends(_api_key_scheme)]) -> None:
        if supplied is not None:
            candidate = supplied.encode()
            # Compare against every key so timing does not reveal which one matched.
            matches = [hmac.compare_digest(candidate, k) for k in keys]
            if any(matches):
                return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    return require_api_key


async def run_checks(
    checks: Mapping[str, ReadinessCheck], timeout_s: float
) -> tuple[bool, dict[str, str]]:
    """Run all checks concurrently in threads; error details stay in the logs only."""

    async def one(name: str, check: ReadinessCheck) -> tuple[str, str]:
        try:
            await asyncio.wait_for(asyncio.to_thread(check), timeout=timeout_s)
        except TimeoutError:
            _log.warning("readiness_check_failed", check=name, reason="timeout")
            return name, "timeout"
        except Exception as exc:  # noqa: BLE001  (any failure means "not ready")
            _log.warning("readiness_check_failed", check=name, error=repr(exc))
            return name, f"error: {type(exc).__name__}"
        return name, "ok"

    results = dict(await asyncio.gather(*(one(n, c) for n, c in checks.items())))
    return all(v == "ok" for v in results.values()), results


def create_service_app(
    *,
    service: str,
    version: str,
    settings: ApiSettings,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    features: Mapping[str, bool] | None = None,
    routers: Sequence[APIRouter] = (),
    on_shutdown: Sequence[Callable[[], None]] = (),
) -> FastAPI:
    """FastAPI app with health/readiness, API-key protected /v1, request IDs and CORS.

    `routers` are mounted under /v1 behind the API key.
    """
    checks = dict(readiness_checks or {})
    feature_flags = dict(features or {})
    if not settings.key_list():
        _log.warning("api_keys_not_configured", service=service, effect="all /v1 requests get 401")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _log.info("service_started", service=service, version=version)
        try:
            yield
        finally:
            for close in on_shutdown:
                close()
            _log.info("service_stopped", service=service)

    app = FastAPI(title=f"iqs-{service}", version=version, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=[API_KEY_HEADER, REQUEST_ID_HEADER, "Content-Type"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if 0 < len(incoming) <= _MAX_REQUEST_ID_LEN else str(uuid7())
        clear_context()
        bind_context(request_id=request_id, service=service)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers[REQUEST_ID_HEADER] = request_id
        _log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=elapsed_ms,
        )
        clear_context()
        return response

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(service=service, version=version)

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        responses={503: {"model": ReadyResponse}},
        tags=["ops"],
    )
    async def ready() -> JSONResponse:
        ok, results = await run_checks(checks, settings.ready_timeout_s)
        body = ReadyResponse(status="ready" if ok else "not_ready", checks=results)
        code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(body.model_dump(), status_code=code)

    v1 = APIRouter(prefix=f"/{API_VERSION}", dependencies=[Depends(api_key_dependency(settings))])

    @v1.get("/capabilities", response_model=CapabilitiesResponse, tags=["meta"])
    def capabilities() -> CapabilitiesResponse:
        st, why = feature_status(feature_flags)
        return CapabilitiesResponse(
            service=service, version=version, features=feature_flags, status=st, reason=why
        )

    for router in routers:
        v1.include_router(router)
    app.include_router(v1)
    return app
