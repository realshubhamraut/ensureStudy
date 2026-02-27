# Page 57: OCR Pipeline Deep Dive — 6 Engines & Hybrid Strategy

---

## 57.1 Overview

ensureStudy implements a **multi-engine OCR pipeline** with 6 different OCR backends, a hybrid fallback strategy, and pre-processing stages for image enhancement. This enables recognition of printed text, handwritten notes, mathematical formulas, and scanned documents.

### Source: 18 files in `backend/ai-service/app/services/`

---

## 57.2 OCR Engine Inventory

| Engine | File | Type | Best For | License |
|--------|------|------|----------|---------|
| Tesseract | `ocr_service.py` | Traditional | Printed text, clean documents | Apache 2.0 |
| TrOCR (HuggingFace) | `ocr_service.py` | Transformer | Handwritten text | MIT |
| Nanonets OCR2 | `nanonets_ocr.py` | VLM (Qwen2.5-VL) | Complex layouts | Open |
| SageMaker OCR | `sagemaker_ocr.py` | Cloud (AWS) | Production scale | Managed |
| Hybrid OCR | `hybrid_ocr.py` | Multi-engine | Best accuracy | Combined |
| EasyOCR | via adapter | Deep learning | Multi-language | Apache 2.0 |

---

## 57.3 OCR Adapter Pattern

### Source: `services/ocr_adapter.py`

```python
class OCRAdapter:
    """Unified interface for multiple OCR engines"""
    
    def __init__(self):
        self.engines = {
            "tesseract": TesseractEngine(),
            "trocr": TrOCREngine(),
            "nanonets": NanonetsEngine(),
        }
        self.default_engine = "hybrid"
    
    def recognize(self, image, engine: str = None) -> OCRResult:
        engine = engine or self.default_engine
        
        if engine == "hybrid":
            return self._hybrid_recognize(image)
        
        return self.engines[engine].recognize(image)
```

---

## 57.4 Hybrid OCR Strategy

### Source: `services/hybrid_ocr.py`

```python
class HybridOCR:
    """
    Multi-engine OCR with confidence-based selection.
    
    Strategy:
    1. Run Tesseract (fast, good for printed text)
    2. If confidence < 0.7, run TrOCR (better for handwritten)
    3. If still < 0.7, run Nanonets (VLM, best accuracy)
    4. Return highest-confidence result
    """
    
    def recognize(self, image) -> OCRResult:
        # Stage 1: Tesseract (fast)
        tess_result = self.tesseract.recognize(image)
        if tess_result.confidence >= 0.7:
            return tess_result
        
        # Stage 2: TrOCR (transformer)
        trocr_result = self.trocr.recognize(image)
        if trocr_result.confidence >= 0.7:
            return trocr_result
        
        # Stage 3: Nanonets VLM (most accurate)
        nano_result = self.nanonets.recognize(image)
        
        # Return best result
        results = [tess_result, trocr_result, nano_result]
        return max(results, key=lambda r: r.confidence)
```

---

## 57.5 Image Pre-Processing Pipeline

### Source: `services/image_enhancer.py`

```python
class ImageEnhancer:
    """Pre-process images for better OCR accuracy"""
    
    def enhance(self, image) -> np.ndarray:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Deskew (fix rotation)
        deskewed = self._deskew(gray)
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(deskewed, h=10)
        
        # 4. Adaptive thresholding (binarization)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Contrast enhancement
        enhanced = self._enhance_contrast(binary)
        
        return enhanced
```

### Enhancement Steps

| Step | Purpose | Technique |
|------|---------|-----------|
| Grayscale | Simplify image | `cv2.cvtColor` |
| Deskew | Fix scanned tilt | Hough transform → rotation |
| Denoise | Remove noise | Non-local means denoising |
| Binarize | Black/white text | Adaptive Gaussian threshold |
| Contrast | Sharpen text | CLAHE histogram equalization |

---

## 57.6 Layout Analysis

### Source: `services/layout_service.py`

```python
class LayoutService:
    """Detect text regions, tables, figures in documents"""
    
    def analyze_layout(self, image) -> LayoutResult:
        # Detect text blocks
        text_regions = self._detect_text_regions(image)
        
        # Detect tables
        tables = self._detect_tables(image)
        
        # Detect figures/diagrams
        figures = self._detect_figures(image)
        
        return LayoutResult(
            text_regions=text_regions,
            tables=tables,
            figures=figures,
            reading_order=self._determine_reading_order(text_regions)
        )
```

---

## 57.7 PDF Processing

### Source: `services/pdf_extractor.py`, `services/pdf_processor.py`

```python
class PDFProcessor:
    def process(self, pdf_path: str) -> ProcessedDocument:
        # 1. Try digital text extraction (PyMuPDF)
        text = self._extract_digital_text(pdf_path)
        
        if text and len(text.strip()) > 100:
            # Digital PDF — no OCR needed
            return ProcessedDocument(text=text, method="digital")
        
        # 2. Convert pages to images
        images = pdf2image.convert_from_path(pdf_path)
        
        # 3. OCR each page
        pages = []
        for i, img in enumerate(images):
            enhanced = self.enhancer.enhance(np.array(img))
            ocr_result = self.hybrid_ocr.recognize(enhanced)
            pages.append(ocr_result.text)
        
        return ProcessedDocument(
            text="\n".join(pages),
            method="ocr",
            page_count=len(pages)
        )
```

---

## 57.8 Searchable PDF Generation

### Source: `services/searchable_pdf.py`

```python
class SearchablePDFService:
    """Convert scanned PDFs to searchable PDFs with invisible text layer"""
    
    def make_searchable(self, input_pdf: str, output_pdf: str):
        # 1. Extract pages as images
        # 2. OCR each page
        # 3. Create invisible text overlay at correct coordinates
        # 4. Merge overlay with original image
        # Output: PDF that looks identical but has selectable text
```

---

## 57.9 LaTeX/Math Formula Recognition

### Source: `services/latex_converter.py`

```python
class LaTeXConverter:
    """Convert detected math regions to LaTeX notation"""
    
    def image_to_latex(self, math_region: np.ndarray) -> str:
        # Use VLM (Nanonets/Gemini) to recognize math formulas
        prompt = "Convert this mathematical formula image to LaTeX notation."
        latex = self.llm.generate(prompt, image=math_region)
        return latex  # e.g., "\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}"
```

---

## 57.10 OCR Performance

| Engine | Speed | Accuracy (Printed) | Accuracy (Handwritten) | GPU Required |
|--------|-------|--------------------|-----------------------|-------------|
| Tesseract | ~100ms/page | 95%+ | 60% | No |
| TrOCR | ~500ms/page | 92% | 85% | Preferred |
| Nanonets VLM | ~2s/page | 98% | 90% | Yes |
| Hybrid | ~500ms-2s | 98% | 90% | Preferred |
