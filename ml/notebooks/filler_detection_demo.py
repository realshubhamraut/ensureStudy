#!/usr/bin/env python3
"""
🎤 Filler Detection Demo Script
================================

Industry-standard speech fluency analysis using:
- Wav2Vec2 pre-trained embeddings (HuggingFace)
- Librosa acoustic feature extraction
- MFCC, Pitch, Spectral analysis

No custom training required - uses pre-trained models!

Run this script to test the filler detection:
    python ml/notebooks/filler_detection_demo.py
"""

import sys
sys.path.insert(0, '/Users/proxim/projects/ensureStudy')

import numpy as np
import librosa
from pathlib import Path

# Import our detector
from backend.ai_service.app.services.filler_detector import get_filler_detector


def demo_text_analysis():
    """Demo text-based filler detection."""
    print("\n" + "="*60)
    print("📝 TEXT-BASED FILLER DETECTION")
    print("="*60)
    
    detector = get_filler_detector()
    
    # Test samples with different filler densities
    samples = [
        {
            "name": "Excellent Speech",
            "text": """
            The key to effective communication is clarity and structure. 
            When presenting ideas, start with your main point, provide 
            supporting evidence, and conclude with a clear summary.
            """
        },
        {
            "name": "Moderate Fillers",
            "text": """
            So, the thing is, we need to focus on, you know, improving 
            our presentation skills. I think it's actually quite important 
            to practice regularly.
            """
        },
        {
            "name": "Heavy Fillers",
            "text": """
            Um, so, like, I was thinking, you know, um, that we should, 
            uh, basically, like, focus on, I mean, uh, the main point is, 
            like, um, we need to, you know, practice more, right?
            """
        }
    ]
    
    for sample in samples:
        result = detector.analyze_fluency(
            transcript=sample["text"],
            duration_seconds=15
        )
        
        print(f"\n🔹 {sample['name']}:")
        print(f"   Score: {result['overall_score']}/100")
        print(f"   Fillers: {result['filler_count']} found")
        print(f"   Types: {result['fillers_by_type']}")
        print(f"   Feedback: {result['feedback']}")


def demo_audio_analysis():
    """Demo audio-based filler detection."""
    print("\n" + "="*60)
    print("🎵 AUDIO-BASED FILLER DETECTION")
    print("="*60)
    
    detector = get_filler_detector()
    
    # Generate synthetic audio samples
    sr = 16000
    
    # 1. Clean vowel sound (potential filler)
    t = np.linspace(0, 0.5, int(0.5 * sr))
    clean_vowel = 0.5 * np.sin(2 * np.pi * 200 * t)  # 200 Hz fundamental
    
    # 2. Variable pitch (less likely filler)
    freq = 200 + 50 * np.sin(2 * np.pi * 5 * t)  # Varying frequency
    variable_sound = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # 3. Noisy signal
    noise = np.random.randn(int(0.5 * sr)) * 0.3
    
    samples = [
        ("Steady Vowel (filler-like)", clean_vowel),
        ("Variable Pitch (speech-like)", variable_sound),
        ("Noise (non-speech)", noise),
    ]
    
    for name, audio in samples:
        result = detector.detect_fillers_from_audio(audio, sr)
        print(f"\n🔹 {name}:")
        print(f"   Filler Likelihood: {result['filler_likelihood']:.1%}")
        print(f"   Likely Type: {result['filler_type']}")
        print(f"   Is Voiced: {result['is_voiced']}")
        print(f"   Clarity Score: {result['clarity_score']}/100")


def demo_wav2vec2_embeddings():
    """Demo Wav2Vec2 embedding extraction."""
    print("\n" + "="*60)
    print("🧠 WAV2VEC2 EMBEDDING EXTRACTION")
    print("="*60)
    
    detector = get_filler_detector()
    
    # Load model and extract embeddings from synthetic audio
    print("\n📥 Loading Wav2Vec2 model...")
    detector.load_models()
    
    # Generate test audio
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    audio = 0.5 * np.sin(2 * np.pi * 200 * t)
    
    embeddings = detector.extract_audio_embeddings(audio, sr)
    
    if embeddings is not None:
        print(f"\n✅ Embeddings extracted!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Device: {detector.device}")
    else:
        print("\n⚠️ Wav2Vec2 not available, using librosa fallback")


def main():
    """Run all demos."""
    print("🎤 INDUSTRY-STANDARD FILLER DETECTION DEMO")
    print("==========================================")
    print("\nThis demo uses:")
    print("  • Wav2Vec2 pre-trained embeddings (HuggingFace)")
    print("  • Librosa acoustic feature extraction")
    print("  • MFCC, Pitch, and Spectral analysis")
    print("  • Zero custom training required!")
    
    # Run demos
    demo_text_analysis()
    demo_audio_analysis()
    demo_wav2vec2_embeddings()
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print("\nTo use in your code:")
    print("```python")
    print("from backend.ai_service.app.services.filler_detector import get_filler_detector")
    print("")
    print("detector = get_filler_detector()")
    print("result = detector.analyze_fluency(transcript, audio, duration)")
    print("print(result['overall_score'], result['feedback'])")
    print("```")


if __name__ == "__main__":
    main()
