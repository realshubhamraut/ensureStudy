# Teacher Materials Ingestion System

## Overview

This system enables teachers to upload PDF/image documents to classrooms. Documents are processed through OCR, chunked, embedded, and indexed for RAG retrieval in the AI Tutor.

## Quick Start

### 1. Start Docker Services

```bash
docker-compose up -d postgres qdrant redis minio
```

### 2. Run Database Migration

```bash
docker exec ensure-study-postgres psql -U ensure_study_user -d ensure_study \
  -f /path/to/003_document_ingestion.sql
```

### 3. Start Application

```bash
./run-local.sh
```

### 4. Upload a Document

```bash
curl -X POST "http://localhost:8001/api/classrooms/YOUR_CLASS_ID/materials/upload" \
  -F "file=@document.pdf" \
  -F "title=Physics Chapter 1"
```

---

## API Endpoints

### Upload Material

```http
POST /api/classrooms/{class_id}/materials/upload
Content-Type: multipart/form-data

file: <PDF or image file>
title: Optional document title
```

**Response:**
```json
{
  "doc_id": "abc123",
  "status": "uploaded",
  "status_url": "/api/classrooms/{class_id}/materials/abc123/status",
  "filename": "document.pdf",
  "file_size": 102400
}
```

### Check Status

```http
GET /api/classrooms/{class_id}/materials/{doc_id}/status
```

**Response:**
```json
{
  "doc_id": "abc123",
  "status": "indexed",
  "page_count": 10,
  "chunk_count": 45,
  "requires_manual_review": false
}
```

### Query Sidebar

```http
GET /api/ai-tutor/documents/{doc_id}/sidebar?query=newton+laws&top_k=5
```

**Response:**
```json
{
  "doc_id": "abc123",
  "title": "Physics Chapter 1",
  "top_matches": [
    {
      "chunk_id": "chunk-1",
      "page_number": 3,
      "bbox": [100, 200, 400, 250],
      "text_snippet": "Newton's third law states...",
      "similarity": 0.89
    }
  ],
  "pdf_url": "http://minio:9000/presigned-url...",
  "version": 1
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | localhost:9000 | MinIO S3 API endpoint |
| `MINIO_ACCESS_KEY` | ensurestudy | MinIO access key |
| `MINIO_SECRET_KEY` | minioadmin123 | MinIO secret key |
| `MINIO_BUCKET` | ensurestudy-documents | S3 bucket name |
| `CELERY_BROKER_URL` | redis://localhost:6379/1 | Celery broker URL |
| `CORE_SERVICE_URL` | http://localhost:8000 | Core service URL |
| `NANONETS_CONFIDENCE_THRESHOLD` | 0.7 | OCR confidence threshold |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Teacher Upload                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI Service (FastAPI)                         │
│   POST /materials/upload → Store to MinIO → Enqueue Job          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Celery Worker                                │
│   Fetch → PDF→Images → OCR → PII Redact → Chunk → Embed → Index │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │   PostgreSQL   │       │    Qdrant     │
            │  (metadata)    │       │  (vectors)    │
            └───────────────┘       └───────────────┘
```

---

## Security

### PII Redaction

All extracted text is automatically scanned and redacted before indexing:
- Email addresses → `[EMAIL_REDACTED]`
- Phone numbers → `[PHONE_REDACTED]`

### Access Control

- Document uploads require classroom membership
- Presigned URLs have 1-hour TTL
- All API calls should validate user permissions

### Data Retention

- Raw files: 30 days (configurable)
- Processed chunks: Same as document retention
- Vectors: Deleted when document is deleted

---

## Running Tests

```bash
# Unit tests
pytest tests/test_document_pipeline.py -v

# Integration tests (requires services)
pytest tests/test_document_pipeline.py -v -m integration
```

---

## Troubleshooting

### MinIO Connection Failed

```bash
# Check MinIO is running
docker ps | grep minio

# Check MinIO console
open http://localhost:9001
# Login: ensurestudy / minioadmin123
```

### OCR Not Working

```bash
# Check Tesseract is installed
tesseract --version

# Install on macOS
brew install tesseract

# Install PDF tools
brew install poppler  # for pdf2image
```

### Celery Worker Not Processing

```bash
# Start worker manually
cd backend/ai-service
celery -A app.workers worker -l info -Q documents,ocr
```
