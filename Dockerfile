FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY uv.lock ./

RUN pip install --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir -e .

CMD ["run_crew"]
