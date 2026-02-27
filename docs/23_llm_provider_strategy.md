# Page 23: LLM Provider Strategy & API Key Management

---

## 23.1 Overview

ensureStudy employs a **multi-provider LLM strategy** that supports OpenAI GPT-4, Google Gemini, Groq (Mixtral/LLaMA), and local models (Mistral-7B via Ollama). This enables cost optimization, fallback resilience, and task-specific model selection.

---

## 23.2 Provider Inventory

| Provider | Models | Use Case | Cost Tier |
|----------|--------|----------|-----------|
| **OpenAI** | GPT-4, GPT-3.5-turbo | Primary tutoring, complex reasoning | $$$ |
| **Google Gemini** | Gemini 1.5 Flash, Gemini 1.5 Pro | Meeting summarization, long-context tasks | $$ |
| **Groq** | Mixtral-8x7B, LLaMA 3 70B | Fast classification, topic extraction | $ |
| **Ollama (local)** | Mistral-7B, LLaMA 3 8B | Assessment generation, offline fallback | Free |
| **OpenAI Whisper** | whisper-1 | Speech-to-text transcription | $ |
| **AWS Polly** | Various voices | Text-to-speech with visemes | $ |

---

## 23.3 LLM Service Architecture

### Source: `backend/ai-service/app/services/llm_service.py`

```python
class LLMService:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai")
        self.model = os.getenv("LLM_MODEL", "gpt-4")
        
    async def generate(self, prompt, system_prompt=None, **kwargs):
        if self.provider == "openai":
            return await self._openai_generate(prompt, system_prompt, **kwargs)
        elif self.provider == "gemini":
            return await self._gemini_generate(prompt, system_prompt, **kwargs)
        elif self.provider == "groq":
            return await self._groq_generate(prompt, system_prompt, **kwargs)
        elif self.provider == "ollama":
            return await self._ollama_generate(prompt, system_prompt, **kwargs)
```

### Provider Selection by Feature

| Feature | Primary Provider | Fallback | Justification |
|---------|-----------------|----------|---------------|
| **Tutor chat** | OpenAI GPT-4 | Gemini 1.5 Flash | Best reasoning quality |
| **Meeting summary** | Gemini 1.5 Flash | GPT-3.5-turbo | Long-context (1M tokens) |
| **Topic extraction** | Groq (Mixtral) | GPT-3.5-turbo | Fast, structured output |
| **Subject classification** | Groq (LLaMA 3) | Local Mistral | Speed-critical |
| **Assessment generation** | Ollama (Mistral-7B) | GPT-3.5-turbo | Volume, cost-free |
| **Question scoring** | GPT-4 | Gemini 1.5 Pro | Accuracy-critical |
| **Curriculum generation** | GPT-4 | Gemini 1.5 Pro | Complex reasoning |
| **Web search analysis** | Groq (Mixtral) | GPT-3.5-turbo | Fast processing |
| **Speech-to-text** | OpenAI Whisper | Local Whisper | Accuracy |
| **Text-to-speech** | AWS Polly | Browser TTS | Viseme support |

---

## 23.4 API Key Management

### Environment Configuration

```bash
# .env
# === Primary LLM ===
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai              # openai, gemini, groq, ollama
LLM_MODEL=gpt-4                  # Model name

# === Google Gemini ===
GOOGLE_API_KEY=AIza...            # For meeting summaries, long-context
GEMINI_MODEL=gemini-1.5-flash

# === Groq (Fast Inference) ===
GROQ_API_KEY=gsk_...              # For classification, extraction
GROQ_MODEL=mixtral-8x7b-32768

# === Local Models ===
OLLAMA_HOST=http://localhost:11434  # Local Ollama server
OLLAMA_MODEL=mistral:7b

# === Speech/Audio ===
# OPENAI_API_KEY is reused for Whisper
AWS_ACCESS_KEY_ID=...             # For AWS Polly TTS
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1

# === Embedding ===
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # Local model
```

### Key Rotation Strategy

1. All API keys stored exclusively in `.env` (gitignored)
2. `.env.production.example` provides template without real values
3. Production keys set via environment variables or secrets manager
4. Keys never logged (startup logger masks with `***`)

---

## 23.5 Cost Optimization

### Token Usage Patterns

| Operation | Avg Input Tokens | Avg Output Tokens | Provider | Monthly Cost (est.) |
|-----------|-----------------|-------------------|----------|-------------------|
| Tutor chat | 2,000 | 500 | GPT-4 | Variable |
| Topic extraction | 500 | 200 | Groq | ~Free tier |
| Assessment gen | 300 | 400 | Ollama | $0 (local) |
| Meeting summary | 5,000 | 1,000 | Gemini | Low |
| Question scoring | 800 | 200 | GPT-4 | Variable |

### Cost Reduction Strategies

| Strategy | Implementation |
|----------|----------------|
| **Response caching** | Redis cache with query hash keys, 1h TTL |
| **Embedding caching** | Redis cache with text hash keys, 7d TTL |
| **Local models first** | Use Ollama for high-volume, low-complexity tasks |
| **Groq for speed** | Free tier covers most classification needs |
| **Gemini for length** | 1M token context handles long transcripts cheaply |
| **Streaming** | SSE streaming reduces perceived latency, same cost |

---

## 23.6 Fallback Chain

```python
async def generate_with_fallback(prompt, providers=None):
    providers = providers or ["openai", "gemini", "groq", "ollama"]
    
    for provider in providers:
        try:
            return await generate(prompt, provider=provider)
        except RateLimitError:
            logger.warning(f"{provider} rate limited, trying next")
            continue
        except APIError as e:
            logger.error(f"{provider} failed: {e}")
            continue
    
    raise AllProvidersFailedError("No LLM provider available")
```

---

## 23.7 Embedding Strategy

| Model | Location | Dimension | Use Case |
|-------|----------|-----------|----------|
| `all-mpnet-base-v2` | Local (HuggingFace) | 768 | Document, notes, meeting embeddings |
| `text-embedding-3-small` | OpenAI API | 1536 | (Configured but backup) |

The primary embedding model runs **locally** via `sentence-transformers`, eliminating per-call API costs for the highest-volume operation (every document chunk, query, and meeting segment).

---

## 23.8 Prompt Engineering Patterns

### System Prompt Structure

```python
TUTOR_SYSTEM_PROMPT = """You are an expert tutor for {subject}.
Student Level: {tal_level} ({level_description})
Topic: {topic}
Classroom: {classroom_name}

Context from study materials:
{rag_context}

Instructions:
- Adapt explanations to the student's assessed level
- Use examples relevant to their curriculum
- Reference the provided context when possible
- Use LaTeX for mathematical expressions
- Be encouraging and supportive"""
```

### JSON-Structured Output

```python
TOPIC_EXTRACTION_PROMPT = """Extract topics from this syllabus text.
Return ONLY valid JSON in this format:
{
    "topics": [
        {"name": "Topic Name", "subtopics": ["Sub1", "Sub2"], "difficulty": "medium"}
    ]
}

Syllabus text:
{text}"""
```
