"""
PDF Generator Service
Combines extracted/enhanced images into a single PDF document

Technologies used:
- img2pdf for lossless image-to-PDF conversion
- Pillow for image processing
"""
import img2pdf
from PIL import Image
from pathlib import Path
import logging
from typing import List, Optional
import io

logger = logging.getLogger(__name__)


class PDFGenerator:
    """
    Generates PDF documents from a collection of images
    """
    
    def __init__(
        self,
        page_size: str = "A4",  # A4, Letter, or (width, height) in mm
        dpi: int = 300,
        margin_mm: float = 10.0
    ):
        self.page_size = page_size
        self.dpi = dpi
        self.margin_mm = margin_mm
        
        # Standard page sizes in mm
        self.PAGE_SIZES = {
            "A4": (210, 297),
            "Letter": (216, 279),
            "Legal": (216, 356)
        }
    
    def images_to_pdf(
        self,
        image_paths: List[str],
        output_path: str,
        title: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Convert a list of images to a single PDF
        
        Args:
            image_paths: List of paths to image files (in order)
            output_path: Path for the output PDF
            title: Optional PDF title metadata
            progress_callback: Optional callback(progress: float, message: str)
        
        Returns:
            Path to the generated PDF
        """
        if not image_paths:
            raise ValueError("No images provided")
        
        logger.info(f"Generating PDF from {len(image_paths)} images")
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process images - ensure they're compatible with img2pdf
        processed_images = []
        
        for i, img_path in enumerate(image_paths):
            if progress_callback:
                progress = (i / len(image_paths)) * 0.8  # 80% for processing
                progress_callback(progress, f"Processing image {i+1}/{len(image_paths)}")
            
            try:
                processed = self._prepare_image(img_path)
                if processed:
                    processed_images.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process image {img_path}: {e}")
        
        if not processed_images:
            raise ValueError("No valid images could be processed")
        
        # Generate PDF
        if progress_callback:
            progress_callback(0.9, "Creating PDF...")
        
        try:
            # Get page size
            if isinstance(self.page_size, str):
                page_w, page_h = self.PAGE_SIZES.get(self.page_size, (210, 297))
            else:
                page_w, page_h = self.page_size
            
            # Convert mm to points (1 inch = 25.4 mm, 1 inch = 72 points)
            page_size_pt = (img2pdf.mm_to_pt(page_w), img2pdf.mm_to_pt(page_h))
            
            # Create layout function
            layout = img2pdf.get_layout_fun(page_size_pt)
            
            # Generate PDF
            pdf_bytes = img2pdf.convert(processed_images, layout_fun=layout)
            
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            
            logger.info(f"PDF generated: {output_path} ({len(processed_images)} pages)")
            
            if progress_callback:
                progress_callback(1.0, "PDF created successfully")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise
    
    def _prepare_image(self, image_path: str) -> Optional[bytes]:
        """
        Prepare an image for PDF conversion
        img2pdf works best with JPEG/PNG bytes
        """
        try:
            with Image.open(image_path) as img:
                # Convert RGBA to RGB if needed (PDF doesn't support alpha)
                if img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Save to bytes buffer as JPEG for smaller file size
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=90)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error preparing image {image_path}: {e}")
            return None
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get basic information about a PDF file"""
        import os
        
        path = Path(pdf_path)
        if not path.exists():
            return {"error": "PDF not found"}
        
        return {
            "path": str(path),
            "filename": path.name,
            "size_bytes": os.path.getsize(pdf_path),
            "size_mb": round(os.path.getsize(pdf_path) / (1024 * 1024), 2)
        }


def generate_pdf_from_images(
    image_paths: List[str],
    output_path: str,
    title: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> str:
    """
    Convenience function to generate PDF from images
    
    Returns:
        Path to the generated PDF
    """
    generator = PDFGenerator()
    return generator.images_to_pdf(image_paths, output_path, title, progress_callback)
