"""
LLM Prompt Templates for AI Tutor

Designed for Mistral-7B-Instruct (decoder-only instruction-following model).
Uses Mistral's [INST]...[/INST] chat format for optimal instruction following.

The LLM:
- ONLY reasons over provided context
- NEVER discovers new knowledge
- ALWAYS cites sources by chunk_id
"""

# ============================================================================
# Mistral Instruction Template
# ============================================================================

MISTRAL_TUTOR_TEMPLATE = """<s>[INST] You are an expert academic tutor specializing in {subject}.

CONTEXT FROM STUDY MATERIALS:
{context}

STUDENT QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY using the provided context above
2. Give a {mode} explanation
3. Use {language_style}
4. Do NOT include inline citations like "[ch_001]" or "Based on the study materials" - sources are shown separately
5. If the context doesn't contain the answer, clearly state: "I don't have information about this in the provided materials."
6. Structure your answer with clear steps or bullet points when appropriate
7. Do NOT add external facts or knowledge not present in the context
8. Write naturally without referencing the context directly - just give the answer

Provide your answer now: [/INST]"""


def build_tutor_prompt(
    question: str,
    context: str,
    subject: str = "General",
    response_mode: str = "short",
    language_style: str = "layman"
) -> str:
    """
    Build prompt for Mistral-7B-Instruct.
    
    Args:
        question: Student's question
        context: Assembled context from MCP (already token-limited)
        subject: Academic subject (for context)
        response_mode: 'short' or 'detailed'
        language_style: 'scientific', 'layman', or 'simple'
        
    Returns:
        Complete prompt string for Mistral-7B-Instruct
        
    The LLM receives ONLY:
    - Selected chunks from retrieval (via context)
    - The student question
    - Clear instructions
    
    The LLM NEVER receives:
    - Entire documents
    - Raw databases
    - User history dump
    """
    mode_instruction = "concise and focused" if response_mode == "short" else "detailed and comprehensive"
    
    # Language style instructions
    language_instructions = {
        "scientific": "professional, scientific language with technical terminology appropriate for academic study",
        "layman": "clear, everyday language that is easy to understand without sacrificing accuracy",
        "simple": "the simplest possible explanation, like explaining to a young student or beginner"
    }
    lang_instruction = language_instructions.get(language_style, language_instructions["layman"])
    
    return MISTRAL_TUTOR_TEMPLATE.format(
        subject=subject,
        context=context,
        question=question,
        mode=mode_instruction,
        language_style=lang_instruction
    )


# ============================================================================
# Prompt Explanation
# ============================================================================

"""
## Prompt Design for Mistral-7B-Instruct

### Why Mistral?
Mistral-7B-Instruct is a decoder-only transformer optimized for:
- Instruction following
- RAG-grounded reasoning
- Structured output generation
- Multi-step reasoning
- Educational QA

### Mistral Instruction Format
Mistral uses the [INST]...[/INST] format:
- <s> = Start of sequence
- [INST] = Start of instruction
- [/INST] = End of instruction (model generates after this)

### Anti-Hallucination Measures
1. "Answer ONLY using the provided context"
2. "Do NOT add external facts"  
3. "If the context doesn't contain the answer, clearly state..."
4. "Cite specific chunk_ids when making claims"

### Advantages over FLAN-T5
- Better at multi-step reasoning
- Stronger instruction following
- More stable JSON output
- Better at educational explanations
- Lower hallucination rate with RAG
"""
