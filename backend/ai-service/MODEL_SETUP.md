# EnsureStudy AI Service - Model Setup Guide

## Overview

This document explains the model architecture and setup for the EnsureStudy AI service.

## Current Model Strategy

We use a **hybrid approach** that combines API calls with local fallbacks:

### 1. Subject Classification & Moderation
**Primary:** HuggingFace Inference API (`facebook/bart-large-mnli`)
**Fallback:** Keyword-based classification

**File:** `app/services/llm_provider.py` → `TextClassifier`

**Why Hybrid?**
- ✅ No local model download needed (avoids memory issues)
- ✅ Fast keyword fallback when API is unavailable
- ✅ Works without API key (with lower accuracy)
- ✅ No bus errors or memory crashes

**Accuracy:**
- With API: ~90-95% accuracy
- Keyword fallback: ~60-70% accuracy (sufficient for most cases)

### 2. Follow-up Question Generation
**Model:** `google/flan-t5-base` via HuggingFace Inference API

**File:** `app/services/followup_generator.py`

**Fallback:** Subject-aware keyword-based questions

### 3. Main Answer Generation
**Primary:** AWS SageMaker (if configured)
**Secondary:** `microsoft/Phi-3-mini-4k-instruct` via HuggingFace API

**File:** `app/services/llm_provider.py` → `HuggingFaceLLM` / `SageMakerLLM`

## Environment Variables

```bash
# Required for API mode
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxx
# OR
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
# OR
HF_TOKEN=hf_xxxxxxxxxxxxx

# Optional: AWS SageMaker
USE_SAGEMAKER=false
SAGEMAKER_ENDPOINT=your-endpoint-name
AWS_REGION=us-east-1
```

## Local Model Setup (Optional - For Deployment)

If you want to use local models instead of APIs (e.g., for deployment):

### Download Models

```bash
cd backend/ai-service
python download_models.py
```

This will download (~4-5 GB):
- `facebook/bart-large-mnli` (1.6 GB) - Classification
- `google/flan-t5-base` (900 MB) - Text generation
- `distilbert-base-uncased` (260 MB) - ABCR routing
- `microsoft/trocr-base-handwritten` (1.3 GB) - OCR

### Switch to Local Models

To use local models, modify `TextClassifier` in `app/services/llm_provider.py`:

```python
# Change from API to local pipeline
self._pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # CPU
)
```

**Note:** Local models require significant RAM (8GB+ recommended).

## Current Issues & Solutions

### Issue: Bus Error (Exit 138)
**Cause:** Local `facebook/bart-large-mnli` model too large for available RAM

**Solution:** Using HuggingFace Inference API with keyword fallback

### Issue: API 410 Gone Error
**Cause:** Some models deprecated from free Inference API

**Solution:** Keyword-based fallback automatically handles this

## Testing

```bash
# Test subject classifier
python test_api_classifier.py

# Test full pipeline
python test_models.py
```

## Production Recommendations

### For Web Deployment (Recommended):
1. ✅ Use HuggingFace Inference API (as currently configured)
2. ✅ Keep keyword fallback enabled
3. ✅ Set `HUGGINGFACE_API_KEY` environment variable
4. ❌ Do NOT use local models (memory intensive)

### For High-Volume/Offline Deployment:
1. Use AWS SageMaker with serverless endpoints
2. Download models to cache: `python download_models.py`
3. Configure GPU instance (e.g., g4dn.xlarge)
4. Set `USE_SAGEMAKER=true`

## Model Selection Criteria

| Aspect | Current Choice | Reason |
|--------|---------------|---------|
| **Subject Detection** | Keyword fallback | No local model crashes, works without API key |
| **Follow-up Questions** | HF API + keyword fallback | Fast, reliable, graceful degradation |
| **Answer Generation** | HF API (Phi-3) | Good quality, free tier available |
| **Moderation** | Keyword-based | Simple, fast, no external dependency |

## Future Improvements

1. **OpenAI Integration:** Add GPT-3.5/4 as primary LLM option
2. **Quantized Models:** Use 4-bit quantized models for local deployment
3. **Model Caching:** Implement response caching to reduce API calls
4. **Ensemble Classification:** Combine keyword + API results for higher accuracy

## Support

For issues or questions, check:
- HuggingFace Status: https://status.huggingface.co/
- Model documentation: https://huggingface.co/models
- Environment setup: `/backend/ai-service/.env.example`
