# --- Build stage (has compilers for native extensions) ---
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /workspace

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

COPY . .
RUN poetry install --no-interaction --no-ansi --only main

# --- Runtime stage (no build tools, ~200 MB smaller) ---
FROM python:3.11-slim AS runtime

WORKDIR /workspace

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /workspace /workspace

EXPOSE 8080

CMD ["python", "-m", "mcp_tool_router.server"]
