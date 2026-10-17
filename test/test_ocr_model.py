#!/usr/bin/env python3
"""
Test script to verify Nanonets-OCR2-3B model works correctly.
Run from the ensureStudy directory with venv activated.
"""
import sys
sys.path.insert(0, 'backend/ai-service')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from PIL import Image
import os

# Create a simple test image with text
def create_test_image():
    """Create a simple test image with some text."""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    draw.text((50, 50), "Hello World", fill='black')
    draw.text((50, 100), "This is a test of OCR", fill='black')
    
    test_path = '/tmp/test_ocr_image.png'
    img.save(test_path)
    return test_path

def main():
    print("=" * 60)
    print("Testing Nanonets-OCR2-3B Model")
    print("=" * 60)
    
    # Check if there's an actual image to test with
    test_images = [
        '/tmp/test_ocr_image.png',  # Generated test image
    ]
    
    # Look for any uploaded notes in the data directory
    data_dir = 'data/notes'
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    test_images.insert(0, os.path.join(root, f))
                    break
            if len(test_images) > 1:
                break
    
    # Create test image if needed
    print("\n1. Creating/Finding test image...")
    test_path = create_test_image()
    print(f"   Test image: {test_path}")
    
    # Import the service
    print("\n2. Importing NanonetsOCRService...")
    try:
        from app.services.nanonets_ocr import NanonetsOCRService
        print("   ✓ Import successful")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return
    
    # Create service
    print("\n3. Creating service instance...")
    service = NanonetsOCRService()
    print(f"   Model ID: {service.model_id}")
    
    # Initialize (load model)
    print("\n4. Loading model (this may take 30-60 seconds on first run)...")
    print("   Downloading model weights if not cached...")
    success = service.initialize()
    
    if not success:
        print("   ✗ Failed to initialize model")
        return
    
    print(f"   ✓ Model loaded on device: {service.device}")
    
    # Run OCR
    print("\n5. Running OCR on test image...")
    text, confidence = service.extract_text(test_path)
    
    print(f"\n   ✓ OCR Complete!")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Extracted text ({len(text)} chars):")
    print("-" * 40)
    print(text)
    print("-" * 40)
    
    # Test with first real image if available
    if len(test_images) > 1:
        real_image = test_images[0]
        print(f"\n6. Testing with real uploaded image: {real_image}")
        text2, conf2 = service.extract_text(real_image)
        print(f"   Confidence: {conf2:.2f}")
        print(f"   Extracted text ({len(text2)} chars):")
        print("-" * 40)
        print(text2[:500] if len(text2) > 500 else text2)
        print("-" * 40)

if __name__ == "__main__":
    main()
