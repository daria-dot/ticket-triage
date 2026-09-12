# Builder: compiles/installs dependencies that need a full toolchain
# (scikit-learn, pandas, psycopg wheels). Nothing from this stage ships.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir .

# Runtime: just the venv and source, no compiler, no root.
FROM python:3.11-slim

RUN groupadd --system triage && useradd --system --gid triage --no-create-home triage

COPY --from=builder /venv /venv
COPY src ./src

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER triage

EXPOSE 8000

CMD ["uvicorn", "triage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
