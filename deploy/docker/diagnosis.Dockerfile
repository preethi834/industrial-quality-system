# Diagnosis service image. Build context is the repository root:
#   make build            (or: docker build -f deploy/docker/diagnosis.Dockerfile .)
# Installs exactly the versions in uv.lock (--frozen) for the iqs-diagnosis package only,
# without dev tools. Runs as a non-root user; port 8002.

ARG PYTHON_IMAGE=python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

FROM ghcr.io/astral-sh/uv:0.13.8@sha256:d6be4539ed178df7972a67a8a5244b9663974ba9d83f0a9fab0619a585fd42d8 AS uv

# ---- build: resolve and install into /opt/venv ----------------------------------
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Dependencies first (cached until uv.lock or a pyproject.toml changes).
# Every workspace member's pyproject.toml is needed for uv to validate the lockfile.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/common/pyproject.toml packages/common/
COPY packages/vision/pyproject.toml packages/vision/
COPY packages/diagnosis/pyproject.toml packages/diagnosis/
COPY packages/mcp_server/pyproject.toml packages/mcp_server/
COPY packages/simulator/pyproject.toml packages/simulator/
COPY packages/dashboard/pyproject.toml packages/dashboard/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package iqs-diagnosis --no-install-workspace

# Then our own code, installed as regular (non-editable) packages.
COPY packages/common/src packages/common/src
COPY packages/diagnosis/src packages/diagnosis/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package iqs-diagnosis --no-editable

# ---- runtime -------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin iqs
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IQS_LOGGING__JSON_OUTPUT=true
USER iqs
EXPOSE 8002
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=2)"]
# Access lines come from our request middleware (JSON, with request_id).
CMD ["uvicorn", "--factory", "iqs_diagnosis.api.main:create_app", "--host", "0.0.0.0", "--port", "8002", "--no-access-log"]
