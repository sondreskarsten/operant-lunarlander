FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends swig build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY tests ./tests
COPY README.md ./

ENTRYPOINT ["python", "-m", "operant_lunarlander.differentiate"]
