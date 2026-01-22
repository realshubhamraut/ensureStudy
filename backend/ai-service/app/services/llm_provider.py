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
    "default": "mistralai/Mistral-7B-Instruct-v0.2",
    "fast": "microsoft/Phi-3-mini-4k-instruct",
    "small": "google/flan-t5-large",
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
    Zero-shot text classifier using Hugging Face
    Used for moderation (no LLM needed, faster)
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load classifier pipeline"""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                
                self._pipeline = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=-1  # CPU
                )
                logger.info("Initialized text classifier")
            except Exception as e:
                logger.error(f"Failed to init classifier: {e}")
                raise
        return self._pipeline
    
    def classify(
        self, 
        text: str, 
        labels: List[str],
        multi_label: bool = False
    ) -> Dict[str, float]:
        """
        Classify text into labels
        
        Args:
            text: Text to classify
            labels: Possible label categories
            multi_label: Allow multiple labels
        
        Returns:
            Dict of label -> score
        """
        try:
            result = self.pipeline(
                text, 
                candidate_labels=labels,
                multi_label=multi_label
            )
            
            return dict(zip(result["labels"], result["scores"]))
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {label: 0.0 for label in labels}


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
