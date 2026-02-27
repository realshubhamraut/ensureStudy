# Page 42: Dockerfile Architecture — Multi-Stage Builds

---

## 42.1 Overview

ensureStudy uses **5 Dockerfiles** (3 development, 2 production) with multi-stage builds, layer caching, non-root users, and health checks. This page documents every Dockerfile line-by-line.

---

## 42.2 Dockerfile Comparison Matrix

| Property | Core Dev | Core Prod | AI Dev | AI Prod | Frontend |
|----------|----------|-----------|--------|---------|----------|
| Base Image | python:3.11-slim | python:3.11-slim | python:3.11-slim | python:3.11-slim | node:20 |
| Stages | 1 | 2 (builder + runtime) | 1 | 2 (builder + runtime) | 1 |
| Server | Flask dev server | Gunicorn (2W, 4T) | Uvicorn | Uvicorn | Next.js |
| Port | 8000 | 8000 | 8001 | 8001 | 3000 |
| User | root | appuser (1000) | root | appuser (1000) | root |
| Healthcheck | curl | Python urllib | curl | Python urllib | — |
| System Deps | gcc, libpq-dev | libpq5 | gcc, libreoffice, tesseract | ffmpeg | — |

---

## 42.3 Core Service — Development (`Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# System deps for psycopg compilation
RUN apt-get update && apt-get install -y \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer (cached if requirements unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app
ENV FLASK_ENV=development
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 42.4 Core Service — Production (`Dockerfile.prod`)

**Multi-stage build** separating build tools from runtime:

```dockerfile
# Stage 1: BUILD — includes gcc, cmake, build-essential
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get install -y build-essential libpq-dev
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user gunicorn

# Stage 2: RUNTIME — minimal, no build tools
FROM python:3.11-slim
WORKDIR /app
RUN apt-get install -y libpq5           # Only runtime lib
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .

# Create directories for file uploads
RUN mkdir -p /app/uploads /app/recordings

# Security: non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Gunicorn: 2 workers, 4 threads, 120s timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", 
     "--threads", "4", "--timeout", "120", "app:create_app()"]
```

---

## 42.5 AI Service — Development (`Dockerfile`)

Includes **LibreOffice** and **Tesseract OCR** for document processing:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc libpq-dev curl \
    libreoffice-writer libreoffice-impress libreoffice-common \  # PPTX/DOCX→PDF
    tesseract-ocr tesseract-ocr-eng \                            # OCR
    fonts-liberation fonts-dejavu-core \                          # Document fonts
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

## 42.6 AI Service — Production (`Dockerfile.prod`)

Multi-stage build with **pre-downloaded Whisper model**:

```dockerfile
# Stage 1: BUILD — includes cmake/boost for dlib compilation
FROM python:3.11-slim as builder
RUN apt-get install -y build-essential cmake libboost-python-dev libboost-system-dev
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: RUNTIME
FROM python:3.11-slim
RUN apt-get install -y ffmpeg            # Required for Whisper audio processing
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .

# PRE-DOWNLOAD Whisper model during build (not runtime)
ARG WHISPER_MODEL=small
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

# Security: non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Longer start-period (60s) for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

## 42.7 Image Size Estimates

| Image | Dev Size | Prod Size | Reduction |
|-------|----------|-----------|-----------|
| Core Service | ~500 MB | ~300 MB | 40% |
| AI Service | ~4 GB | ~3 GB | 25% |
| Frontend | ~500 MB | ~200 MB | 60% |

The AI Service image is large due to PyTorch (~2 GB), sentence-transformers, and Whisper model weights.

---

## 42.8 Docker Build Best Practices Used

| Practice | Implementation |
|----------|---------------|
| **Layer caching** | `COPY requirements.txt` before `COPY .` |
| **Multi-stage builds** | Separate builder and runtime stages |
| **Non-root user** | `appuser` (UID 1000) in production |
| **No cache** | `pip install --no-cache-dir` |
| **Apt cleanup** | `rm -rf /var/lib/apt/lists/*` |
| **Health checks** | HTTP endpoint polling |
| **Build args** | `WHISPER_MODEL` configurable at build time |
| **Unbuffered Python** | `PYTHONUNBUFFERED=1` for log visibility |
