"""
Unified LLM Provider with SageMaker + HuggingFace Fallback

Supports:
- SageMaker Serverless (production)
- HuggingFace Inference API (fallback/development)

Environment:
- USE_SAGEMAKER: Enable SageMaker (default: false)
- SAGEMAKER_ENDPOINT: Endpoint name
- AWS_REGION: AWS region (default: us-east-1)
"""
import os
import logging
import time
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration
USE_SAGEMAKER = os.getenv("USE_SAGEMAKER", "false").lower() == "true"
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_COLD_TIMEOUT = int(os.getenv("SAGEMAKER_COLD_TIMEOUT", "30"))

# Available HuggingFace models (fallback)
MODELS = {
    "default": "microsoft/Phi-3-mini-4k-instruct",  # More reliable on HF Inference API
    "fast": "microsoft/Phi-3-mini-4k-instruct",
    "small": "google/flan-t5-large",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",  # May have availability issues
}


class HuggingFaceLLM:
    """
    LLM provider using Hugging Face Inference API
    Works with free tier API key
    """
    
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model_name = model_name or MODELS["default"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            logger.warning("No HUGGINGFACE_API_KEY found. LLM calls will fail.")
        
        self._client = None
    
    @property
    def client(self):
        """Lazy load the HF client"""
        if self._client is None:
            try:
                from langchain_huggingface import HuggingFaceEndpoint
                
                self._client = HuggingFaceEndpoint(
                    repo_id=self.model_name,
                    huggingfacehub_api_token=self.api_key,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                    task="text-generation",  # Explicitly set task type
                )
                logger.info(f"Initialized HuggingFace LLM: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize HF LLM: {e}")
                raise
        return self._client
    
    def invoke(self, prompt: str) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        try:
            response = self.client.invoke(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"LLM invoke error: {e}")
            raise
    
    async def ainvoke(self, prompt: str) -> str:
        """Async version of invoke"""
        try:
            response = await self.client.ainvoke(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"LLM async invoke error: {e}")
            raise
    
    def generate_structured(
        self, 
        prompt: str, 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output
        
        Args:
            prompt: Input prompt with JSON instructions
            schema: Expected output schema (for validation)
        
        Returns:
            Parsed JSON dict
        """
        import json
        
        # Enhance prompt for JSON output
        json_prompt = f"""{prompt}

IMPORTANT: Return ONLY valid JSON, no other text. Format:
{json.dumps(schema, indent=2)}"""
        
        response = self.invoke(json_prompt)
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {response[:200]}")
            return {}


class TextClassifier:
    """
    Zero-shot text classifier using Groq API with Llama 70B
    
    Uses groq cloud API with llama-3.3-70b-versatile for high accuracy
    Falls back to distilbert only if Groq fails
    """
    
    def __init__(self, api_key: str = None):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        self._pipeline = None
        self._groq_client = None
        self._model_name = "typeform/distilbert-base-uncased-mnli"  # Fallback only
        
        if self.groq_api_key:
            logger.info("✅ Groq API key found - using llama-3.3-70b-versatile for classification")
        else:
            logger.warning("⚠️ No GROQ_API_KEY - will use local distilbert fallback")
    
    @property
    def groq_client(self):
        """Lazy load Groq client"""
        if self._groq_client is None and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
                logger.info("✅ Groq client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client: {e}")
        return self._groq_client
    
    @property
    def pipeline(self):
        """Lazy load the local classification pipeline (FALLBACK ONLY)"""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                device = -1  # CPU only
                logger.info(f"Loading LOCAL fallback model: {self._model_name}...")
                
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model=self._model_name,
                    device=device
                )
                logger.info(f"✅ Successfully loaded LOCAL fallback model")
                
            except Exception as e:
                logger.error(f"❌ Failed to load local model: {e}")
                raise RuntimeError(f"Cannot load local classification model: {e}")
        
        return self._pipeline
    
    def _classify_with_groq(
        self,
        text: str,
        labels: List[str]
    ) -> Dict[str, float]:
        """
        Use Groq API with Llama 70B for classification
        """
        try:
            # Create a prompt for classification
            labels_str = ", ".join(labels)
            prompt = f"""You are a subject classifier for educational questions. 

Analyze this question and determine which subject it belongs to: {labels_str}

Question: "{text}"

Think about the key concepts and terminology in the question. Consider:
- Physics questions often involve forces, motion, energy, electricity, magnetism, optics, waves
- Biology questions involve living things, cells, DNA, evolution, anatomy, ecology
- Chemistry questions involve elements, compounds, reactions, molecules, acids, bases
- Mathematics questions involve numbers, equations, calculations, geometry, algebra

Respond with ONLY a JSON object containing confidence scores for each subject. The scores must sum to 1.0.

Example format:
{{"physics": 0.85, "biology": 0.05, "chemistry": 0.05, "mathematics": 0.05}}

Your response (JSON only):"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"Groq raw response: {result_text}")
            
            # Parse JSON response
            import json
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            scores = json.loads(result_text)
            
            # Ensure all labels have scores
            for label in labels:
                if label not in scores:
                    scores[label] = 0.0
            
            logger.info(f"✅ Groq classification: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"❌ Groq classification failed: {e}")
            raise
    
    def classify(
        self, 
        text: str, 
        labels: List[str],
        multi_label: bool = False
    ) -> Dict[str, float]:
        """
        Classify text into labels using Groq API (70B model) with local fallback
        
        Args:
            text: Text to classify
            labels: Possible label categories
            multi_label: Allow multiple labels
        
        Returns:
            Dict of label -> score
        """
        try:
            # Try Groq API first (much better accuracy with 70B model)
            if self.groq_client:
                try:
                    return self._classify_with_groq(text, labels)
                except Exception as groq_error:
                    logger.warning(f"⚠️ Groq failed, falling back to local model: {groq_error}")
            
            # Fallback to local model
            logger.info("Using local distilbert fallback")
            hypothesis_template = "This is a question about {}."
            
            result = self.pipeline(
                text, 
                candidate_labels=labels,
                hypothesis_template=hypothesis_template,
                multi_label=multi_label
            )
            
            scores = dict(zip(result["labels"], result["scores"]))
            logger.debug(f"Local classification scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            raise


class SageMakerLLM:
    """
    LLM provider using AWS SageMaker Serverless.
    
    Falls back to HuggingFace API on cold start or errors.
    """
    
    def __init__(
        self,
        endpoint_name: str = SAGEMAKER_ENDPOINT,
        region: str = AWS_REGION,
        fallback_model: str = None
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.fallback_model = fallback_model or MODELS["default"]
        
        self._client = None
        self._hf_fallback = None
        
        if not endpoint_name:
            logger.warning("No SAGEMAKER_ENDPOINT configured, will use HuggingFace")
    
    @property
    def sagemaker_client(self):
        """Lazy load SageMaker runtime client."""
        if self._client is None and self.endpoint_name:
            try:
                import boto3
                self._client = boto3.client(
                    "sagemaker-runtime",
                    region_name=self.region
                )
                logger.info(f"Initialized SageMaker client: {self.endpoint_name}")
            except Exception as e:
                logger.error(f"Failed to init SageMaker: {e}")
        return self._client
    
    @property
    def hf_fallback(self):
        """Get HuggingFace fallback LLM."""
        if self._hf_fallback is None:
            self._hf_fallback = HuggingFaceLLM(model_name=self.fallback_model)
        return self._hf_fallback
    
    def invoke(self, prompt: str) -> str:
        """
        Generate text using SageMaker with HuggingFace fallback.
        """
        import json
        
        # Try SageMaker first
        if self.sagemaker_client:
            try:
                start = time.time()
                
                response = self.sagemaker_client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps({
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1024,
                            "temperature": 0.7,
                            "do_sample": True
                        }
                    })
                )
                
                result = json.loads(response["Body"].read().decode())
                elapsed = time.time() - start
                
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    text = result.get("generated_text", "")
                else:
                    text = str(result)
                
                logger.info(f"SageMaker response in {elapsed:.1f}s")
                return text
                
            except Exception as e:
                error_str = str(e)
                
                # Check for cold start timeout
                if "Unable to locate credentials" in error_str:
                    logger.error("AWS credentials not configured")
                elif "ModelError" in error_str or "timeout" in error_str.lower():
                    logger.warning(f"SageMaker cold/error, falling back: {e}")
                else:
                    logger.error(f"SageMaker error: {e}")
        
        # Fallback to HuggingFace
        logger.info("Using HuggingFace fallback")
        return self.hf_fallback.invoke(prompt)
    
    async def ainvoke(self, prompt: str) -> str:
        """Async version - uses sync invoke in thread."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, prompt)


class SearchQueryExtractor:
    """
    LLM-powered search query extractor using Groq API.
    
    Replaces hardcoded keyword lists with intelligent extraction.
    Handles acronyms (AC, DC, pH), scientific terms, and context.
    """
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_client = None
        
        if self.groq_api_key:
            logger.info("✅ SearchQueryExtractor initialized with Groq API")
        else:
            logger.warning("⚠️ No GROQ_API_KEY - will use original query as fallback")
    
    @property
    def groq_client(self):
        """Lazy load Groq client"""
        if self._groq_client is None and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client: {e}")
        return self._groq_client
    
    def extract_search_query(
        self,
        question: str,
        subject: str = None,
        conversation_history: list = None
    ) -> str:
        """
        Extract optimal search query from user's educational question.
        
        Args:
            question: User's question
            subject: Detected subject (physics, chemistry, etc.)
            conversation_history: Previous messages for context
            
        Returns:
            Optimized search query for web/Wikipedia/PDF search
        """
        # Try LLM extraction first
        if self.groq_client:
            try:
                return self._extract_with_groq(question, subject, conversation_history)
            except Exception as e:
                logger.warning(f"⚠️ Groq extraction failed: {e}")
        
        # Fallback: use original query (search APIs handle natural language)
        return self._simple_fallback(question)
    
    def _extract_with_groq(
        self,
        question: str,
        subject: str = None,
        conversation_history: list = None
    ) -> str:
        """Use Groq LLM to extract search terms."""
        import time
        start = time.time()
        
        # Build context from conversation history if available
        context_hint = ""
        if conversation_history and len(conversation_history) > 0:
            # Get the last assistant response for context
            for msg in reversed(conversation_history):
                if msg.get('role') == 'assistant':
                    prev_response = msg.get('content', '')[:200]
                    context_hint = f"\nPrevious context: {prev_response}"
                    break
        
        prompt = f"""Extract the optimal web search query from this educational question.

Question: "{question}"
Subject: {subject or 'general'}{context_hint}

Rules:
1. PRESERVE important terms: acronyms (AC, DC, DNA, pH, UV, IR), names (Newton, Einstein), formulas
2. REMOVE generic words: explain, differentiate, compare, what is, tell me, describe, how does
3. Output 2-6 key search terms that would work well for Wikipedia/Google
4. If the question mentions comparing two things (A vs B), include BOTH terms
5. Keep the terms in a natural search order

Examples:
- "Differentiate between AC and DC" → "AC DC alternating direct current"
- "What is Ohm's law?" → "Ohm's law electricity"
- "Explain the process of photosynthesis" → "photosynthesis"
- "Why are electric motors preferred over heat engines?" → "electric motors vs heat engines advantages"

Output ONLY the search query, no explanation or quotes."""

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent output
            max_tokens=100
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up the result (remove quotes if present)
        result = result.strip('"\'')
        
        elapsed = (time.time() - start) * 1000
        logger.info(f"[LLM-EXTRACT] ✅ '{question[:40]}...' → '{result}' ({elapsed:.0f}ms)")
        
        return result
    
    def _simple_fallback(self, question: str) -> str:
        """
        Simple fallback: remove common question words and return.
        This is used when LLM is unavailable.
        """
        import re
        
        # Remove common question prefixes
        patterns = [
            r'^(what is |what are |explain |describe |how does |how do |why is |why are |tell me about |differentiate between |difference between |compare |contrast )',
        ]
        
        result = question.lower()
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # Clean up punctuation but keep important chars
        result = re.sub(r'[?!.,;:]', '', result)
        result = result.strip()
        
        logger.info(f"[FALLBACK-EXTRACT] '{question[:40]}...' → '{result}'")
        return result if result else question


# ============================================================================
# Singleton instances
# ============================================================================

@lru_cache(maxsize=1)
def get_llm(model: str = "default"):
    """
    Get cached LLM instance.
    
    Uses SageMaker if USE_SAGEMAKER=true, otherwise HuggingFace.
    """
    if USE_SAGEMAKER and SAGEMAKER_ENDPOINT:
        logger.info(f"Using SageMaker LLM: {SAGEMAKER_ENDPOINT}")
        return SageMakerLLM(endpoint_name=SAGEMAKER_ENDPOINT)
    else:
        model_name = MODELS.get(model, model)
        logger.info(f"Using HuggingFace LLM: {model_name}")
        return HuggingFaceLLM(model_name=model_name)


@lru_cache(maxsize=1)
def get_classifier() -> TextClassifier:
    """Get cached classifier instance"""
    return TextClassifier()


@lru_cache(maxsize=1)
def get_search_extractor() -> SearchQueryExtractor:
    """Get cached search query extractor instance"""
    return SearchQueryExtractor()
