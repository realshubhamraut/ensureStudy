"""
LLM Reasoning Service using Mistral-7B-Instruct

Supports TWO modes:
1. HuggingFace Inference API (for local testing, no model download)
2. Local model loading (for cloud deployment)

Set LLM_USE_API=true in .env to use API mode (recommended for testing)
"""
import os
import json
import time
import logging
from typing import Optional

from ..api.schemas.tutor import ResponseMode
from ..utils.prompts import build_tutor_prompt
from ..config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Response Structure
# ============================================================================

class LLMResponse:
    """Parsed LLM response."""
    
    def __init__(
        self,
        answer_short: str,
        answer_detailed: Optional[str],
        confidence: float,
        reasoning: str,
        suggested_topics: list,
        raw_response: str,
        generation_time_ms: int
    ):
        self.answer_short = answer_short
        self.answer_detailed = answer_detailed
        self.confidence = confidence
        self.reasoning = reasoning
        self.suggested_topics = suggested_topics
        self.raw_response = raw_response
        self.generation_time_ms = generation_time_ms


# ============================================================================
# HuggingFace Inference API (For Local Testing - No Download)
# ============================================================================

_hf_client = None


def get_hf_api_client():
    """Get HuggingFace Inference API client."""
    global _hf_client
    
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set. Get one free at https://huggingface.co/settings/tokens")
        
        _hf_client = InferenceClient(
            model=settings.LLM_MODEL,
            token=api_key
        )
        logger.info(f"Using HuggingFace API for: {settings.LLM_MODEL}")
    
    return _hf_client


def call_mistral_api(prompt: str) -> tuple:
    """
    Call Mistral via HuggingFace Inference API.
    No model download required - runs on HF servers.
    
    Uses chat_completion for better compatibility with instruction models.
    
    Args:
        prompt: Complete prompt with [INST]...[/INST] format
        
    Returns:
        (response_text, generation_time_ms)
    """
    client = get_hf_api_client()
    
    start = time.time()
    
    try:
        # Use chat_completion for instruction-tuned models
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            top_p=0.95,
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        elapsed_ms = int((time.time() - start) * 1000)
        print(f"[LLM] HuggingFace API responded in {elapsed_ms}ms")
        
        # Debug: show response preview
        response_preview = response_text[:200].replace('\n', ' ') if response_text else 'EMPTY'
        print(f"[LLM] Response preview: {response_preview}...")
        
        return response_text.strip(), elapsed_ms
        
    except Exception as e:
        logger.error(f"HuggingFace API error: {e}")
        print(f"[LLM] HuggingFace error: {e}")
        raise


# ============================================================================
# Local Model Loading (For Cloud Deployment)
# ============================================================================

_model = None
_tokenizer = None


def get_mistral_local():
    """
    Lazy-load Mistral-7B-Instruct model locally.
    Requires ~14GB RAM. Use for cloud deployment.
    """
    global _model, _tokenizer
    
    if _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Loading local LLM model: {settings.LLM_MODEL}")
        print(f"Loading LLM model locally: {settings.LLM_MODEL}")
        
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL,
            trust_remote_code=True
        )
        
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        # Determine device and dtype
        if settings.LLM_USE_GPU:
            import torch
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
        
        _model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        logger.info(f"Mistral loaded locally on {device_map}")
    
    return _model, _tokenizer


def call_mistral_local(prompt: str) -> tuple:
    """
    Call Mistral locally for answer generation.
    
    Args:
        prompt: Complete prompt with [INST]...[/INST] format
        
    Returns:
        (response_text, generation_time_ms)
    """
    import torch
    
    model, tokenizer = get_mistral_local()
    
    start = time.time()
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
        padding=True
    )
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    )
    
    elapsed_ms = int((time.time() - start) * 1000)
    
    return response.strip(), elapsed_ms


# ============================================================================
# Unified Call Function (Auto-selects API or Local)
# ============================================================================

def call_mistral(prompt: str) -> tuple:
    """
    Call Mistral model - automatically chooses API or local.
    
    Set LLM_USE_API=true in .env to use HuggingFace API (recommended for testing)
    Set LLM_USE_API=false for local model (recommended for deployment)
    """
    use_api = os.getenv("LLM_USE_API", "true").lower() == "true"
    
    if use_api:
        logger.info("Using HuggingFace Inference API")
        return call_mistral_api(prompt)
    else:
        logger.info("Using local Mistral model")
        return call_mistral_local(prompt)


# ============================================================================
# Main Reasoning Function
# ============================================================================

def generate_answer(
    question: str,
    context: str,
    subject: str = "General",
    response_mode: ResponseMode = ResponseMode.SHORT,
    language_style: str = "layman"
) -> LLMResponse:
    """
    Generate answer using Mistral-7B-Instruct.
    
    Uses HuggingFace API by default (no model download).
    Set LLM_USE_API=false to use local model.
    """
    prompt = build_tutor_prompt(
        question=question,
        context=context,
        subject=subject,
        response_mode=response_mode.value,
        language_style=language_style
    )
    
    try:
        raw_response, gen_time = call_mistral(prompt)
        logger.info(f"Mistral generated response in {gen_time}ms")
        print(f"[LLM] ✅ Mistral responded in {gen_time}ms")
    except Exception as e:
        logger.error(f"Mistral generation error: {e}")
        print(f"[LLM] ❌ Error: {e}")
        if settings.DEBUG:
            print(f"[LLM] ⚠ Using mock response (DEBUG mode)")
            raw_response = _get_mock_response(question, context)
            gen_time = 100
        else:
            raise e
    
    parsed = _parse_response(raw_response, response_mode)
    
    return LLMResponse(
        answer_short=parsed.get("answer_short", raw_response[:500]),
        answer_detailed=parsed.get("answer_detailed") if response_mode == ResponseMode.DETAILED else None,
        confidence=parsed.get("confidence", 0.7),
        reasoning=parsed.get("reasoning", "Generated from context"),
        suggested_topics=parsed.get("suggested_topics", []),
        raw_response=raw_response,
        generation_time_ms=gen_time
    )


def _parse_response(response: str, mode: ResponseMode) -> dict:
    """Parse LLM response into structured format."""
    try:
        if "{" in response and "}" in response:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    answer = response.strip()
    
    # Generate genuinely short answer for short mode (2-3 sentences)
    if mode == ResponseMode.SHORT:
        # Extract first complete paragraph or first 2-3 sentences
        paragraphs = answer.split('\n\n')
        short_answer = paragraphs[0] if paragraphs else answer
        
        # If still too long, get first 2-3 sentences
        sentences = short_answer.replace('\n', ' ').split('. ')
        if len(sentences) > 3:
            short_answer = '. '.join(sentences[:3]) + '.'
        
        # Clean up
        short_answer = short_answer.strip()
        if len(short_answer) > 400:
            # Last resort: find a good break point
            short_answer = short_answer[:400].rsplit('.', 1)[0] + '.'
    else:
        short_answer = answer
    
    confidence = 0.7
    if len(answer) > 100:
        confidence = 0.8
    if any(x in answer.lower() for x in ["step", "process", "method", "example"]):
        confidence = 0.85
    if len(answer) > 300 and any(x in answer for x in ["$$", "**", "##"]):
        confidence = 0.9
    
    return {
        "answer_short": short_answer,
        "answer_detailed": answer,  # Always include full answer for toggling
        "confidence": confidence,
        "reasoning": "Answer derived from provided context",
        "suggested_topics": []
    }


def _get_mock_response(question: str, context: str) -> str:
    """Mock response for development when API/model fails."""
    if "photosynthesis" in question.lower():
        return """Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.

## The Two Main Stages

1. **Light-dependent reactions**: Occur in the thylakoid membranes, producing ATP and NADPH

2. **Calvin Cycle (Light-independent)**: Uses ATP and NADPH to fix $CO_2$ into glucose in the stroma

## The Overall Equation

$$6CO_2 + 6H_2O + \\text{light} \\rightarrow C_6H_{12}O_6 + 6O_2$$

Or in words: Six carbon dioxide molecules plus six water molecules, with light energy, produce one glucose molecule and six oxygen molecules."""
    
    return f"""Here's what I found about your question:

This topic is covered in the study materials. Please ensure you have a valid `HUGGINGFACE_API_KEY` set in your `.env` file for real AI responses.

*This is a fallback response - check the server logs for errors.*"""

