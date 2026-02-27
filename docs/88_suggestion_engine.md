# Page 88: Suggestion Engine & Follow-up Generator

> Dynamic "Students Also Ask" system with context-aware, non-repetitive follow-up question suggestions using phrase extraction, template generation, diversity filtering, and anti-recursion.

---

## 88.1 Architecture

```mermaid\nflowchart TB\n    INPUT[\"❓ User Question + RAG Context Chunks\"] --> PE[\"PhraseExtractor<br/>10.7KB — TF-IDF + NER key phrases\"]\n    PE --> SE\n\n    subgraph SE[\"SuggestionEngine — 22KB, 622 lines\"]\n        direction TB\n        S1[\"extract_topic()<br/>Main topic from question\"]\n        S2[\"generate_candidates()<br/>Template-based generation\"]\n        S3[\"filter_duplicates()<br/>Hash + recursion detection\"]\n        S4[\"score_candidates()<br/>50% question + 40% chunk + 10% recency\"]\n        S5[\"apply_diversity_filter()<br/>Greedy, reject cosine > 0.7\"]\n        S1 --> S2 --> S3 --> S4 --> S5\n    end\n\n    SE --> FG[\"FollowupGenerator<br/>7.9KB — LLM-based follow-ups\"]\n    FG --> OUT[\"📋 4 Suggested Questions\"]\n\n    style SE fill:#3b82f6,color:#fff\n    style FG fill:#f59e0b,color:#000\n```

### Source Files

| File | Size | Role |
|------|------|------|
| `services/suggestion_engine.py` | 22KB | Core suggestion pipeline |
| `services/suggestion_templates.py` | 6.3KB | Question templates by intent |
| `services/followup_generator.py` | 7.9KB | LLM-based follow-up questions |
| `services/phrase_extractor.py` | 10.7KB | Key phrase extraction |

---

## 88.2 SuggestionEngine

### Data Models

```python
@dataclass
class SuggestedQuestion:
    id: str
    text: str              # "What are the applications of Newton's Third Law?"
    intent: str            # "application", "comparison", "cause_effect"
    score: float           # Relevance score
    novel: bool            # Not previously shown
    source_phrases: List[str]
    action: str = "query"  # action type
    embedding: Optional[List[float]] = None

@dataclass
class SuggestionHistory:
    hash: str              # SHA-256 of normalized text
    text: str
    shown_at: str          # ISO timestamp
```

### Main Pipeline

```python
def generate_suggestions(
    self,
    user_question: str,
    answer: str,
    context_chunks: List[dict],
    session_history: List[str] = None,    # Previously shown
    session_resources: List[str] = None,  # Session resource phrases
    canonical_seed: str = None,           # Immutable topic anchor
    k: int = None                         # Number of suggestions
) -> List[SuggestedQuestion]
```

### 5-Stage Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Extract | `_extract_main_topic()` | Get topic from "tell me about X" → "X" |
| 2. Generate | `_generate_candidates()` | Template-based: "What are the applications of {topic}?" |
| 3. Filter | `_filter_duplicates()` | Hash + anti-recursion check |
| 4. Score | `_score_candidates()` | Weighted: 50% question sim + 40% chunk sim + 10% recency |
| 5. Diversify | `_apply_diversity_filter()` | Greedy selection, reject cosine sim > 0.7 |

### Anti-Recursion Protection

```python
# CRITICAL: Prevents "What are the causes of What Were The Causes Of..."
# Detects when a suggestion text appears inside another suggestion
def _filter_duplicates(self, candidates, session_history):
    for candidate in candidates:
        normalized = candidate.text.lower()
        # Check for nested repetition
        words = normalized.split()
        for i in range(2, len(words)):
            if words[i:i+3] == words[0:3]:  # Repeated prefix
                reject(candidate)
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `N_SUGGESTIONS` | 4 | Default suggestions per query |
| `DIVERSITY_THRESHOLD` | 0.7 | Max cosine sim between suggestions |
| `CHUNK_SIM_WEIGHT` | 0.4 | Weight for context relevance |
| `SESSION_RECENCY_WEIGHT` | 0.1 | Weight for session recency |
| `SUGGEST_MAX_PHRASES` | 8 | Max phrases per extraction |
| `SUGGEST_HISTORY_LIMIT` | 50 | LRU history size |

---

## 88.3 Suggestion Templates

### Source: `services/suggestion_templates.py` (6.3KB)

Templates categorized by **intent**:

| Intent | Template Example |
|--------|-----------------|
| `definition` | "What exactly is {phrase}?" |
| `comparison` | "How does {phrase} compare to {related}?" |
| `application` | "What are the real-world applications of {phrase}?" |
| `cause_effect` | "What causes {phrase}?" |
| `example` | "Can you give an example of {phrase}?" |
| `deep_dive` | "Explain {phrase} in more detail" |
| `timeline` | "What is the history of {phrase}?" |
| `pros_cons` | "What are the advantages and disadvantages of {phrase}?" |

---

## 88.4 FollowupGenerator

### Source: `services/followup_generator.py` (7.9KB)

LLM-based approach for generating natural follow-up questions:

```python
class FollowupGenerator:
    def generate(self, question, answer, context, k=3):
        """Uses Groq LLM to generate k natural follow-up questions
        that a student would realistically ask next"""
```
