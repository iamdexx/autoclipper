FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY autoclipper ./autoclipper
RUN pip install --no-cache-dir -e ".[llm]"
COPY config.yaml ./
VOLUME ["/app/data"]
CMD ["autoclipper", "watch"]
