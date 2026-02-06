"""
AWS Polly TTS Service
Provides high-quality text-to-speech with viseme data for lip sync.

AWS Polly Free Tier: 5M characters/month for 12 months
Neural voices provide the best quality for interviews.
"""

# Load .env file for AWS credentials
import os
from pathlib import Path
from dotenv import load_dotenv

# First try AI service's own .env
ai_env = Path(__file__).resolve().parents[2] / '.env'
print(f"[Polly] Looking for AI .env at: {ai_env} (exists: {ai_env.exists()})")
if ai_env.exists():
    load_dotenv(ai_env, override=True)
    print(f"[Polly] Loaded AI .env, AWS_ACCESS_KEY_ID present: {bool(os.getenv('AWS_ACCESS_KEY_ID'))}")
# Then try root .env  
root_env = Path(__file__).resolve().parents[4] / '.env'
print(f"[Polly] Looking for root .env at: {root_env} (exists: {root_env.exists()})")
if root_env.exists():
    load_dotenv(root_env, override=False)

# boto3 is optional - service will work without it (fallback to browser TTS)
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    BOTO3_AVAILABLE = False

import json
import os
import hashlib
import base64
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PollyService:
    """AWS Polly TTS service with viseme support for avatar lip sync."""
    
    # Available neural voices for interviews
    VOICES = {
        'male': {
            'id': 'Matthew',
            'engine': 'neural',
            'language': 'en-US'
        },
        'female': {
            'id': 'Joanna', 
            'engine': 'neural',
            'language': 'en-US'
        }
    }
    
    # Polly viseme to Oculus viseme mapping
    # Polly uses ARPAbet-based visemes, TalkingHead uses Oculus visemes
    VISEME_MAP = {
        'p': 'PP',    # p, b, m
        't': 'DD',    # t, d, n
        'S': 'CH',    # sh, zh
        'T': 'TH',    # th
        'f': 'FF',    # f, v
        'k': 'kk',    # k, g, ng
        'i': 'I',     # ee
        'r': 'RR',    # r
        's': 'SS',    # s, z
        'u': 'U',     # oo
        '@': 'aa',    # schwa, uh
        'a': 'aa',    # ah
        'e': 'E',     # eh
        'E': 'E',     # ay
        'o': 'O',     # oh
        'O': 'O',     # aw
        'sil': 'sil'  # silence
    }
    
    def __init__(self):
        """Initialize AWS Polly client."""
        self.client = None
        self._init_client()
        
        # Cache directory for audio files
        self.cache_dir = Path('/tmp/polly_cache')
        self.cache_dir.mkdir(exist_ok=True)
    
    def _init_client(self):
        """Initialize boto3 Polly client with credentials."""
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not installed. AWS Polly TTS will not work. Install with: pip install boto3")
            return
            
        try:
            # Get credentials from environment
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            region = os.getenv('AWS_REGION', 'us-east-1')
            
            if not access_key or not secret_key:
                logger.warning("AWS credentials not configured. Polly TTS will not work.")
                return
            
            self.client = boto3.client(
                'polly',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            logger.info("AWS Polly client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Polly client: {e}")
            self.client = None
    
    def _get_cache_key(self, text: str, voice_id: str) -> str:
        """Generate cache key for text+voice combination."""
        content = f"{text}:{voice_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached audio and visemes if available."""
        audio_path = self.cache_dir / f"{cache_key}.mp3"
        meta_path = self.cache_dir / f"{cache_key}.json"
        
        if audio_path.exists() and meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            return {
                'audio_base64': base64.b64encode(audio_data).decode('utf-8'),
                'visemes': meta.get('visemes', []),
                'voice': meta.get('voice', 'Joanna'),
                'duration_ms': meta.get('duration_ms', 0)
            }
        return None
    
    def _cache_result(self, cache_key: str, audio_data: bytes, meta: Dict[str, Any]):
        """Cache audio and metadata for future use."""
        audio_path = self.cache_dir / f"{cache_key}.mp3"
        meta_path = self.cache_dir / f"{cache_key}.json"
        
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    async def synthesize(
        self, 
        text: str, 
        voice_type: str = 'female'
    ) -> Dict[str, Any]:
        """
        Synthesize speech with viseme data for lip sync.
        
        Args:
            text: Text to synthesize
            voice_type: 'male' or 'female'
            
        Returns:
            {
                'audio_base64': base64 encoded MP3 audio,
                'visemes': [{'time': ms, 'value': 'viseme_id'}, ...]
            }
        """
        if not self.client:
            raise RuntimeError("AWS Polly not configured. Please set AWS credentials.")
        
        voice_config = self.VOICES.get(voice_type, self.VOICES['female'])
        voice_id = voice_config['id']
        engine = voice_config['engine']
        
        # Check cache first
        cache_key = self._get_cache_key(text, voice_id)
        cached = self._get_cached(cache_key)
        if cached:
            logger.info(f"Using cached Polly response for: {text[:50]}...")
            return cached
        
        try:
            # Step 1: Get audio stream
            audio_response = self.client.synthesize_speech(
                Text=text,
                VoiceId=voice_id,
                Engine=engine,
                OutputFormat='mp3',
                LanguageCode=voice_config['language']
            )
            
            audio_data = audio_response['AudioStream'].read()
            
            # Step 2: Get viseme marks
            viseme_response = self.client.synthesize_speech(
                Text=text,
                VoiceId=voice_id,
                Engine=engine,
                OutputFormat='json',
                SpeechMarkTypes=['viseme'],
                LanguageCode=voice_config['language']
            )
            
            # Parse viseme JSON lines
            viseme_data = viseme_response['AudioStream'].read().decode('utf-8')
            visemes = []
            
            for line in viseme_data.strip().split('\n'):
                if line:
                    mark = json.loads(line)
                    if mark.get('type') == 'viseme':
                        # Convert Polly viseme to Oculus format
                        polly_viseme = mark.get('value', 'sil')
                        oculus_viseme = self.VISEME_MAP.get(polly_viseme, 'sil')
                        
                        visemes.append({
                            'time': mark.get('time', 0),
                            'value': oculus_viseme
                        })
            
            # Cache the result
            result = {
                'audio_base64': base64.b64encode(audio_data).decode('utf-8'),
                'visemes': visemes,
                'voice': voice_id,
                'duration_ms': visemes[-1]['time'] if visemes else 0
            }
            
            self._cache_result(cache_key, audio_data, {
                'visemes': visemes,
                'voice': voice_id,
                'duration_ms': visemes[-1]['time'] if visemes else 0
            })
            logger.info(f"Synthesized {len(text)} chars with {len(visemes)} visemes")
            
            return result
            
        except Exception as e:
            logger.error(f"Polly synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Polly service is available."""
        return self.client is not None


# Singleton instance
_polly_service: Optional[PollyService] = None

def get_polly_service() -> PollyService:
    """Get or create Polly service singleton."""
    global _polly_service
    if _polly_service is None:
        _polly_service = PollyService()
    return _polly_service
