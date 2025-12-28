"""
Unified LLM Provider using Hugging Face
Provides LLM access for all agents without requiring OpenAI/GPT-4

Uses Hugging Face Inference API with your HUGGINGFACE_API_KEY
"""
import os
import logging
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Available models (free tier compatible)
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


# Singleton instances
@lru_cache(maxsize=1)
def get_llm(model: str = "default") -> HuggingFaceLLM:
    """Get cached LLM instance"""
    model_name = MODELS.get(model, model)
    return HuggingFaceLLM(model_name=model_name)


@lru_cache(maxsize=1)
def get_classifier() -> TextClassifier:
    """Get cached classifier instance"""
    return TextClassifier()
