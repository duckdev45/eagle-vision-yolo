FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    VISION_BUNDLE_DIR=/app/bundle \
    VISION_FEEDBACK_DB=/data/feedback.sqlite \
    PYTHONUNBUFFERED=1

COPY service/pyproject.toml service/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY service/vision_api ./vision_api
COPY models/service/v43 ./bundle

RUN useradd --system --uid 10001 vision && mkdir /data && chown vision:vision /data
USER vision

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "vision_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
