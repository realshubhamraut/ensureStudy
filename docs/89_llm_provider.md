# Page 89: LLM Provider & API Key Management

> Multi-provider LLM abstraction (HuggingFace, SageMaker, Groq), zero-shot text classification, search query extraction, and rotating API key management with failure recovery.

---

## 89.1 LLM Provider Architecture

```mermaid\nflowchart TB\n    APP[\"🏗️ Application Layer<br/>Agents, Services\"] -->|\"invoke() / ainvoke()\"| LLM\n\n    subgraph LLM[\"LLM Provider Layer\"]\n        direction LR\n        HF[\"HuggingFaceLLM<br/>Dev / fallback\"]\n        SM[\"SageMakerLLM<br/>Production (AWS)\"]\n        TC[\"TextClassifier<br/>Zero-shot\"]\n        SQE[\"SearchQueryExtractor<br/>Smart query extraction\"]\n    end\n\n    LLM -->|\"API calls\"| AKM[\"🔑 APIKeyManager<br/>Round-robin rotation<br/>Failure cooldown 60s\"]\n\n    style HF fill:#3b82f6,color:#fff\n    style SM fill:#f59e0b,color:#000\n    style TC fill:#10b981,color:#fff\n    style AKM fill:#ef4444,color:#fff\n```

### Source Files

| File | Size | Role |
|------|------|------|
| `services/llm_provider.py` | 20.6KB, 582 lines | Multi-provider LLM |
| `services/api_key_manager.py` | 9KB, 269 lines | Key rotation |

---

## 89.2 HuggingFaceLLM

```python
MODEL_CONFIGS = {
    "default": "meta-llama/Llama-3.2-3B-Instruct",
    "fast": "microsoft/Phi-3-mini-4k-instruct",
    "small": "google/flan-t5-large",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

class HuggingFaceLLM:
    def __init__(self, model_name=None, api_key=None,
                 temperature=0.7, max_tokens=1024):
    
    def invoke(self, prompt: str) -> str:
        """Sync text generation"""
    
    async def ainvoke(self, prompt: str) -> str:
        """Async text generation"""
    
    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        """Generate JSON output matching schema"""
```

---

## 89.3 SageMakerLLM

```python
class SageMakerLLM:
    """SageMaker Serverless endpoint with HuggingFace fallback."""
    
    def __init__(self, endpoint_name="ensurestudy-llm-serverless",
                 region="us-east-1", fallback_model=None):
    
    def invoke(self, prompt: str) -> str:
        """
        1. Try SageMaker endpoint
        2. If cold start / error → fall back to HuggingFaceLLM
        """
    
    async def ainvoke(self, prompt: str) -> str:
        """Async via thread executor"""
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SAGEMAKER` | `false` | Enable SageMaker |
| `SAGEMAKER_ENDPOINT` | `ensurestudy-llm-serverless` | AWS endpoint name |
| `AWS_REGION` | `us-east-1` | AWS region |
| `HUGGINGFACE_API_KEY` | — | HF API key |

---

## 89.4 TextClassifier

Zero-shot classification using Groq API with local fallback:

```python
class TextClassifier:
    def classify(self, text: str, labels: List[str], multi_label=False) -> Dict:
        """
        Primary: Groq API with llama-3.3-70b-versatile
        Fallback: Local distilbert pipeline
        
        Returns: {"label_1": 0.85, "label_2": 0.12, ...}
        """
```

---

## 89.5 SearchQueryExtractor

```python
class SearchQueryExtractor:
    """LLM-powered search query extraction.
    
    Replaces hardcoded keyword lists with intelligent extraction.
    Handles: acronyms (AC, DC, pH), scientific terms, context.
    
    "What is the role of ATP in cellular respiration?"
    → "ATP cellular respiration role function"
    """
    
    def extract(self, question: str, subject: str = None,
                conversation_history: list = None) -> List[str]:
        """Returns list of search-optimized query strings"""
    
    def _simple_fallback(self, question: str) -> List[str]:
        """Remove stop words, return keywords (no LLM needed)"""
```

---

## 89.6 APIKeyManager

Thread-safe singleton with rotating key support:

```python
class APIKeyManager:
    FAILURE_COOLDOWN = 60    # Seconds before retrying failed key
    MAX_FAILURES = 5         # Permanent disable threshold
    
    # Load keys from env: GROQ_API_KEY="key1,key2,key3"
    def get_key(self, service_name: str) -> Optional[str]:
        """Round-robin rotation, skipping failed/cooling-down keys"""
    
    def mark_failed(self, service_name: str, key: str, reason: str):
        """
        Increment fail_count
        If fail_count >= MAX_FAILURES → permanently disable
        Otherwise → cooldown for 60 seconds
        """
    
    def reset_key(self, service_name: str, key: str):
        """Reset failure state after successful call"""
    
    def get_stats(self) -> Dict:
        """Per-service: active_keys, disabled_keys, total_calls, fails"""
```

### Key State Tracking

```python
@dataclass
class KeyState:
    key: str
    use_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    last_failed: float = 0.0
    is_disabled: bool = False
```

### Convenience Functions

```python
from services.api_key_manager import get_key, mark_key_failed, reset_key

key = get_key("GROQ_API_KEY")  # Next available key
mark_key_failed("GROQ_API_KEY", key, "rate limited")
```
