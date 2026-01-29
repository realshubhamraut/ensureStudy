"""
Download all required models for ensureStudy AI service

This script downloads models to cache so they're available offline.
Run this before deploying or when setting up the environment.
"""
import os
import sys

print("=" * 70)
print("DOWNLOADING MODELS FOR ENSURESTUDYS AI SERVICE")
print("=" * 70)

# List of models to download
MODELS = [
    {
        "name": "facebook/bart-large-mnli",
        "type": "zero-shot-classification",
        "usage": "Subject detection and moderation",
        "size": "~1.6 GB"
    },
    {
        "name": "google/flan-t5-base",
        "type": "text2text-generation",
        "usage": "Follow-up questions and Q&A generation",
        "size": "~900 MB"
    },
    {
        "name": "distilbert-base-uncased",
        "type": "tokenizer+model",
        "usage": "ABCR (Attention-Based Context Routing)",
        "size": "~260 MB"
    },
    {
        "name": "microsoft/trocr-base-handwritten",
        "type": "ocr",
        "usage": "Handwritten text recognition (OCR)",
        "size": "~1.3 GB"
    }
]

def download_model(model_info):
    """Download a single model"""
    name = model_info["name"]
    model_type = model_info["type"]
    
    print(f"\n{'=' * 70}")
    print(f"Model: {name}")
    print(f"Type: {model_type}")
    print(f"Usage: {model_info['usage']}")
    print(f"Size: {model_info['size']}")
    print(f"{'=' * 70}")
    
    try:
        if model_type == "zero-shot-classification":
            from transformers import pipeline
            print("Downloading zero-shot classification model...")
            model = pipeline("zero-shot-classification", model=name, device=-1)
            print(f"✅ Successfully downloaded {name}")
            
        elif model_type == "text2text-generation":
            from transformers import pipeline
            print("Downloading text generation model...")
            model = pipeline("text2text-generation", model=name, device=-1)
            print(f"✅ Successfully downloaded {name}")
            
        elif model_type == "tokenizer+model":
            from transformers import AutoTokenizer, AutoModel
            print("Downloading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            print(f"✅ Successfully downloaded {name}")
            
        elif model_type == "ocr":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            print("Downloading OCR processor and model...")
            processor = TrOCRProcessor.from_pretrained(name)
            model = VisionEncoderDecoderModel.from_pretrained(name)
            print(f"✅ Successfully downloaded {name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
        return False


def main():
    print("\nThis will download approximately 4-5 GB of models.")
    print("Make sure you have:")
    print("  1. Stable internet connection")
    print("  2. At least 10 GB free disk space")
    print("  3. transformers library installed (pip install transformers)")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print("\n🚀 Starting download...\n")
    
    success_count = 0
    failed_models = []
    
    for model_info in MODELS:
        if download_model(model_info):
            success_count += 1
        else:
            failed_models.append(model_info["name"])
    
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"✅ Successfully downloaded: {success_count}/{len(MODELS)} models")
    
    if failed_models:
        print(f"❌ Failed downloads:")
        for model in failed_models:
            print(f"   - {model}")
    else:
        print("🎉 All models downloaded successfully!")
    
    print(f"\nModels are cached in: {os.path.expanduser('~/.cache/huggingface')}")
    print("You can now run the application offline (for these models).")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
