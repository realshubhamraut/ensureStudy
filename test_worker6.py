#!/usr/bin/env python3
"""Test Worker-6/7 processing"""
import sys
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

# Simulate fetched page
test_page = {
    'url': 'https://en.wikipedia.org/wiki/Photosynthesis',
    'content': '''
    <html><body>
    <h1>Photosynthesis</h1>
    <p>Photosynthesis is the process used by plants to convert light energy into chemical energy.
    It takes place in chloroplasts using chlorophyll. The overall equation is 6CO2 + 6H2O → C6H12O6 + 6O2.
    This process is essential for life on Earth as it produces oxygen and organic compounds.</p>
    </body></html>
    ''',
    'status_code': 200
}

print("\n[TEST-W6] Testing Worker-6 processing...")
print(f"[TEST-W6] Content length: {len(test_page['content'])}")

try:
    # Extract with trafilatura
    import trafilatura
    clean_text = trafilatura.extract(test_page['content'])
    print(f"[TEST-W6] Trafilatura extracted: {len(clean_text) if clean_text else 0} chars")
    print(f"[TEST-W6] Content: {clean_text}")
    
    if clean_text and len(clean_text) >= 100:
        print("[TEST-W6] ✅ Extraction successful!")
    else:
        print("[TEST-W6] ❌ Extraction too short")
        
except Exception as e:
    print(f"[TEST-W6] ❌ Error: {e}")
    import traceback
    traceback.print_exc()
