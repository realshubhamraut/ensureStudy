# MinIO Object Storage Schema

MinIO provides S3-compatible object storage for files, recordings, and artifacts.

## Connection

```python
from minio import Minio

# Development
client = Minio(
    "localhost:9000",
    access_key="minioadmin",  # Or from env
    secret_key="minioadmin",
    secure=False
)

# Docker network
client = Minio(
    "minio:9000",
    access_key=os.getenv("MINIO_ROOT_USER"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    secure=False
)
```

---

## Buckets

### 1. `ensure-study-documents`

**Purpose:** Uploaded classroom materials and student notes.

**Structure:**
```
ensure-study-documents/
├── classrooms/
│   └── {classroom_id}/
│       ├── materials/
│       │   └── {document_id}.pdf
│       └── thumbnails/
│           └── {document_id}_thumb.png
├── students/
│   └── {student_id}/
│       └── notes/
│           └── {note_id}.pdf
└── temp/
    └── {upload_id}/
```

---

### 2. `ensure-study-recordings`

**Purpose:** Meeting recordings and processed chunks.

**Structure:**
```
ensure-study-recordings/
├── meetings/
│   └── {meeting_id}/
│       ├── raw/
│       │   └── recording.webm
│       ├── chunks/
│       │   ├── chunk_0.webm
│       │   └── chunk_1.webm
│       └── processed/
│           └── audio.mp3
└── exams/
    └── {exam_session_id}/
        └── proctor_recording.webm
```

---

### 3. `ensure-study-ocr`

**Purpose:** OCR processing artifacts.

**Structure:**
```
ensure-study-ocr/
└── {document_id}/
    ├── pages/
    │   ├── page_1.png
    │   └── page_2.png
    ├── json/
    │   ├── page_1.json
    │   └── page_2.json
    └── quality/
        └── report.json
```

---

### 4. `ensure-study-mlflow`

**Purpose:** MLflow model artifacts.

**Structure:**
```
ensure-study-mlflow/
└── {experiment_id}/
    └── {run_id}/
        ├── model/
        │   ├── model.pkl
        │   └── requirements.txt
        └── artifacts/
            └── metrics.json
```

---

## Environment Variables

```env
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_ENDPOINT=localhost:9000

# S3-Compatible
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
AWS_S3_ENDPOINT=http://localhost:9000
AWS_S3_BUCKET=ensure-study-documents
```

---

## Bucket Policies

### Public Read (for thumbnails)
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::ensure-study-documents/*/thumbnails/*"]
        }
    ]
}
```

### Authenticated Access (default)
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": ["arn:aws:iam:::user/ensure-study"]},
            "Action": ["s3:*"],
            "Resource": ["arn:aws:s3:::ensure-study-*/*"]
        }
    ]
}
```

---

## Setup Commands

```bash
# Create buckets using mc CLI
mc alias set local http://localhost:9000 minioadmin minioadmin123

mc mb local/ensure-study-documents
mc mb local/ensure-study-recordings
mc mb local/ensure-study-ocr
mc mb local/ensure-study-mlflow

# Set public policy for thumbnails
mc anonymous set download local/ensure-study-documents/classrooms/*/thumbnails/
```
