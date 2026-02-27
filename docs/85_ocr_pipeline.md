# Page 85: OCR Multi-Backend Pipeline

> Full document digitization pipeline: image enhancement → layout detection → multi-backend OCR (TrOCR, SageMaker, EasyOCR) → searchable PDF generation.

---

## 85.1 Pipeline Architecture

```mermaid\nflowchart TB\n    INPUT[\"📄 Input Image/PDF\"] --> IE[\"ImageEnhancer<br/>19KB — contrast, denoise,<br/>deskew, binarize\"]\n    IE --> LS[\"LayoutService<br/>12KB — region detection,<br/>column analysis\"]\n    LS --> OA[\"OCR Adapter<br/>(config-driven)\"]\n\n    OA -->|\"OCR_ADAPTER=trocr\"| TR[\"TrOCRAdapter<br/>microsoft/trocr-base-handwritten<br/>Best for handwriting\"]\n    OA -->|\"OCR_ADAPTER=sagemaker\"| SM[\"SageMakerAdapter<br/>Nanonets-OCR2-3B<br/>High-quality printed\"]\n    OA -->|\"OCR_ADAPTER=easyocr\"| EO[\"EasyOCRAdapter<br/>80+ languages<br/>Multi-language fallback\"]\n\n    TR & SM & EO --> HO[\"HybridOCRService<br/>Tesseract layout + TrOCR line-by-line\"]\n    HO --> SP[\"SearchablePDF<br/>6.7KB — overlay text on scanned pages\"]\n\n    style TR fill:#3b82f6,color:#fff\n    style SM fill:#f59e0b,color:#000\n    style EO fill:#10b981,color:#fff\n```

### Source Files

| File | Size | Role |
|------|------|------|
| `services/image_enhancer.py` | 19KB | Image preprocessing |
| `services/layout_service.py` | 12KB | Document layout analysis |
| `services/ocr_adapter.py` | 17KB | Abstract adapter + 3 backends |
| `services/hybrid_ocr.py` | 12KB | Hybrid Tesseract+TrOCR |
| `services/ocr_service.py` | 16KB | High-level OCR orchestration |
| `services/nanonets_ocr.py` | 9KB | Nanonets API integration |
| `services/sagemaker_ocr.py` | 13KB | AWS SageMaker endpoint |
| `services/searchable_pdf.py` | 7KB | Searchable PDF generation |
| `services/latex_converter.py` | 12KB | LaTeX ↔ text conversion |

---

## 85.2 OCR Adapter Pattern

### Abstract Base

```python
class OCRAdapter(ABC):
    @abstractmethod
    def extract_lines(self, image_bytes: bytes) -> OCRPageResult: ...
    
    @abstractmethod
    def get_model_name(self) -> str: ...

@dataclass
class OCRPageResult:
    lines: List[OCRLine]         # Individual text lines
    full_text: str               # Concatenated text
    avg_confidence: float        # 0-1
    model_used: str              # e.g. "trocr-base-handwritten"
    processing_time_ms: int
```

### Backend Implementations

| Adapter | Model | Best For | Config |
|---------|-------|----------|--------|
| `TrOCRAdapter` | `microsoft/trocr-base-handwritten` | Handwritten notes | `OCR_ADAPTER=trocr` |
| `SageMakerAdapter` | `Nanonets-OCR2-3B` | High-quality printed | `OCR_ADAPTER=sagemaker` |
| `EasyOCRAdapter` | EasyOCR (80+ langs) | Multi-language | `OCR_ADAPTER=easyocr` |

### Factory Function

```python
def get_ocr_adapter(config: dict = None) -> OCRAdapter:
    adapter_type = config.get("OCR_ADAPTER", os.getenv("OCR_ADAPTER_TYPE", "trocr"))
    if adapter_type == "sagemaker":
        return SageMakerAdapter()
    elif adapter_type == "easyocr":
        return EasyOCRAdapter(languages=config.get("EASYOCR_LANGUAGES", "en").split(","))
    else:
        return TrOCRAdapter(model_size=config.get("TROCR_MODEL_SIZE", "base"))
```

---

## 85.3 TrOCR Line Detection

Uses horizontal projection profile to detect text lines, then recognizes each line individually:

```python
class TrOCRAdapter:
    def _detect_text_lines(self, image_np) -> List[Tuple]:
        """Horizontal projection → find gaps → segment lines"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        projection = np.sum(binary, axis=1)
        # Find row ranges where projection > threshold → text lines
    
    def extract_lines(self, image_bytes: bytes) -> OCRPageResult:
        lines = self._detect_text_lines(image_np)
        for line in lines:
            crop = image[y_start:y_end, x_start:x_end]
            pixel_values = self.processor(crop, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values, output_scores=True)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
```

---

## 85.4 HybridOCRService

Combines Tesseract for layout analysis with TrOCR for line recognition:

```python
class HybridOCRService:
    TROCR_MODEL = "microsoft/trocr-base-handwritten"
    
    def extract_text(self, image, use_hybrid=True):
        """
        Returns: (full_text, avg_confidence, lines)
        
        Hybrid approach:
        1. Tesseract _detect_lines() → bounding boxes
        2. TrOCR _recognize_lines_trocr() → text per line
        3. Fallback: Tesseract full-page if TrOCR fails
        """
    
    def _detect_lines(self, img) -> List[TextLine]:
        """Tesseract layout analysis with pytesseract.image_to_data()"""
    
    def _recognize_lines_trocr(self, img, lines) -> List[TextLine]:
        """Parallel TrOCR recognition via HuggingFace API"""
```

---

## 85.5 Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_ADAPTER_TYPE` | `trocr` | Active backend |
| `TROCR_MODEL_SIZE` | `base` | `base` or `large` |
| `HUGGINGFACE_API_KEY` | — | For HF Inference API |
| `SAGEMAKER_OCR_ENABLED` | `false` | Enable SageMaker |
| `SAGEMAKER_OCR_ENDPOINT` | `ensurestudy-ocr-serverless` | AWS endpoint |
| `EASYOCR_LANGUAGES` | `en` | Comma-separated codes |
