FROM python:3.11-slim

LABEL maintainer="SQLens"
LABEL description="SQLens — AI SQL Analyzer | Meta PyTorch x HuggingFace x Scaler Hackathon"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./pyproject.toml
COPY models.py      ./models.py
COPY tasks.py       ./tasks.py
COPY inference.py   ./inference.py
COPY validate.py    ./validate.py
COPY client.py      ./client.py
COPY openenv.yaml   ./openenv.yaml
COPY server/        ./server/

RUN adduser --disabled-password --gecos "" --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
