FROM python:3.12-slim

# Create user first
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/

# Change ownership of /app to the app user
RUN chown -R app:app /app

# Switch to app user before installing dependencies
USER app

# Now install dependencies as the app user
RUN uv sync --no-dev

EXPOSE 8000 7860

ENV PYTHONPATH=/app/src
ENV NOPIN_CONFIG=/app/configs/default.toml

CMD ["uv", "run", "uvicorn", "nopin.api:app", "--host", "0.0.0.0", "--port", "8000"]
