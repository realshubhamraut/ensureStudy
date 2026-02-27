# Page 38: Error Handling, Resilience & Graceful Degradation

---

## 38.1 Overview

ensureStudy implements **defensive error handling** across all services, using try-catch wrappers, graceful fallbacks, optional dependency loading, and structured error responses. This ensures the platform remains functional even when individual AI components fail.

---

## 38.2 API Error Response Format

### Core Service (Flask)

```python
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error.description)}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Authentication required"}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({"error": "Access denied"}), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500
```

### AI Service (FastAPI)

```python
from fastapi import HTTPException

# Structured error responses
raise HTTPException(
    status_code=422,
    detail={
        "error": "Processing failed",
        "message": "Could not extract text from PDF",
        "suggestion": "Ensure the file is a valid PDF"
    }
)
```

---

## 38.3 LLM Fallback Chain

The most critical resilience pattern — ensures tutoring continues even if a provider is down:

```python
FALLBACK_ORDER = ["openai", "gemini", "groq", "ollama"]

async def generate_with_fallback(prompt, **kwargs):
    errors = []
    
    for provider in FALLBACK_ORDER:
        try:
            response = await generate(prompt, provider=provider, **kwargs)
            if provider != FALLBACK_ORDER[0]:
                logger.info(f"Used fallback provider: {provider}")
            return response
            
        except RateLimitError as e:
            logger.warning(f"{provider} rate limited: {e}")
            errors.append((provider, "rate_limit"))
            continue
            
        except TimeoutError as e:
            logger.warning(f"{provider} timeout: {e}")
            errors.append((provider, "timeout"))
            continue
            
        except APIError as e:
            logger.error(f"{provider} API error: {e}")
            errors.append((provider, str(e)))
            continue
    
    # All providers failed
    logger.critical(f"All LLM providers failed: {errors}")
    raise AllProvidersFailedError(errors)
```

---

## 38.4 Optional Dependency Loading

Many AI components gracefully handle missing dependencies:

```python
# DeepFace — optional, graceful fallback
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not installed, face verification disabled")

# Audio detection — optional
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Cassandra — optional analytics storage
try:
    from cassandra.cluster import Cluster
    CASSANDRA_AVAILABLE = True
except ImportError:
    CASSANDRA_AVAILABLE = False
```

### Components with Graceful Fallbacks

| Component | Primary | Fallback | Degradation |
|-----------|---------|----------|-------------|
| Face verification | DeepFace | Face detection only | No identity verification |
| Audio detection | PyAudio | None | Skip audio analysis |
| Speaker diarization | simple-diarizer | Single-speaker mode | No speaker labels |
| OCR | Tesseract + EasyOCR + Surya | Text extraction only | Skip handwritten text |
| Cassandra analytics | Cassandra | Skip storage | No meeting analytics |
| Qdrant embeddings | Qdrant client | Log and skip | No vector search |

---

## 38.5 HTTP Request Error Handling

### AI Service Middleware

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)")
        return response
    except Exception as e:
        duration = time.time() - start
        logger.error(f"{request.method} {request.url.path} → ERROR ({duration:.2f}s): {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )
```

---

## 38.6 Database Connection Resilience

### Connection Pool Configuration

```python
# SQLAlchemy connection pooling with pre-ping
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,           # Test connection before use
    pool_size=5,                  # Base pool size
    max_overflow=10,              # Additional connections
    pool_recycle=300,             # Recycle connections after 5 min
)
```

### Redis Connection Handling

```python
try:
    redis_client = Redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info("Redis connected")
except ConnectionError:
    redis_client = None
    logger.warning("Redis unavailable, caching disabled")
```

---

## 38.7 Document Processing Error Recovery

Each stage in the 7-stage pipeline has independent error handling:

```python
async def process_document(file_path: str) -> dict:
    result = {"status": "processing", "stages": {}}
    
    # Stage 1: Validate
    try:
        validated = validate_file(file_path)
        result["stages"]["validate"] = "success"
    except ValidationError as e:
        return {"status": "failed", "error": f"Validation: {e}"}
    
    # Stage 3: OCR (non-fatal, skip if fails)
    try:
        ocr_text = run_ocr(file_path)
        result["stages"]["ocr"] = "success"
    except OCRError as e:
        logger.warning(f"OCR failed, continuing without: {e}")
        ocr_text = ""
        result["stages"]["ocr"] = "skipped"
    
    # Stage 6: Embedding (non-fatal, retry)
    for attempt in range(3):
        try:
            await embed_chunks(chunks)
            result["stages"]["embedding"] = "success"
            break
        except QdrantError as e:
            logger.warning(f"Embedding attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    else:
        result["stages"]["embedding"] = "failed"
    
    return result
```

---

## 38.8 Proctoring Error Isolation

```python
class ProctorSession:
    def analyze_frame(self, frame):
        detections = {}
        
        for name, detector in self.detectors.items():
            try:
                detections[name] = detector.detect(frame)
            except Exception as e:
                logger.error(f"Detector {name} failed: {e}")
                detections[name] = None  # Skip this detector
        
        # Scoring works with whatever detectors succeeded
        return self.scorer.calculate(detections)
```

---

## 38.9 Frontend Error Boundaries

```typescript
// React Error Boundary for graceful UI failures
class ErrorBoundary extends React.Component {
    componentDidCatch(error, errorInfo) {
        console.error('Component error:', error, errorInfo);
    }
    
    render() {
        if (this.state.hasError) {
            return <ErrorFallback message="Something went wrong" />;
        }
        return this.props.children;
    }
}

// API call wrapper with retry
async function fetchWithRetry(url, options, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url, options);
            if (res.ok) return res.json();
        } catch (e) {
            if (i === retries - 1) throw e;
            await new Promise(r => setTimeout(r, 1000 * (i + 1)));
        }
    }
}
```
