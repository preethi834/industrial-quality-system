# Dashboard image (Streamlit, port 8501). Build context is the repository root:
#   make build   (or: docker build -f deploy/docker/dashboard.Dockerfile .)
# Same pinned base images as the services; installs only iqs-dashboard (+ iqs-common)
# from uv.lock. The dashboard talks to the APIs over HTTP only.

ARG PYTHON_IMAGE=python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

FROM ghcr.io/astral-sh/uv:0.13.8@sha256:d6be4539ed178df7972a67a8a5244b9663974ba9d83f0a9fab0619a585fd42d8 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock .python-version ./
COPY packages/common/pyproject.toml packages/common/
COPY packages/vision/pyproject.toml packages/vision/
COPY packages/diagnosis/pyproject.toml packages/diagnosis/
COPY packages/mcp_server/pyproject.toml packages/mcp_server/
COPY packages/simulator/pyproject.toml packages/simulator/
COPY packages/dashboard/pyproject.toml packages/dashboard/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package iqs-dashboard --no-install-workspace
COPY packages/common/src packages/common/src
COPY packages/dashboard/src packages/dashboard/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package iqs-dashboard --no-editable

FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --user-group --create-home --shell /usr/sbin/nologin iqs
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501
USER iqs
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"]
CMD ["sh", "-c", "exec streamlit run \"$(python -c 'import iqs_dashboard, os; print(os.path.join(os.path.dirname(iqs_dashboard.__file__), \"app.py\"))')\""]
