FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN uv sync --no-dev

RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000 7860

ENV PYTHONPATH=/app/src
ENV NOPIN_CONFIG=/app/configs/default.toml

CMD ["uv", "run", "uvicorn", "nopin.api:app", "--host", "0.0.0.0", "--port", "8000"]