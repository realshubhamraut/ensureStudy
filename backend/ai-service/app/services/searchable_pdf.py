"""
Searchable PDF Generator
Converts images to a searchable PDF with embedded OCR text layer

The OCR text is invisible (behind images) but makes the PDF searchable.
Uses ocrmypdf for OCR processing and img2pdf for initial conversion.
"""
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Callable
import img2pdf
from PIL import Image

logger = logging.getLogger(__name__)


class SearchablePDFGenerator:
    """
    Generates searchable PDFs from images
    
    Pipeline:
    1. Convert images to PDF (img2pdf)
    2. Run OCR to add text layer (ocrmypdf)
    3. Output: PDF with invisible searchable text
    """
    
    def __init__(
        self,
        language: str = "eng",
        deskew: bool = False,  # Disabled - can distort text
        clean: bool = False,   # Disabled - requires 'unpaper' to be installed
        optimize: int = 1
    ):
        """
        Args:
            language: OCR language (eng, hin, etc.)
            deskew: Auto-correct skewed pages (disabled by default - can distort)
            clean: Clean up background noise (disabled - requires 'unpaper')
            optimize: PDF optimization level (0-3)
        """
        self.language = language
        self.deskew = deskew
        self.clean = clean
        self.optimize = optimize
    
    def generate(
        self,
        image_paths: List[str],
        output_path: str,
        title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Generate a searchable PDF from images
        
        Args:
            image_paths: List of image file paths
            output_path: Output PDF path
            title: PDF title metadata
            progress_callback: Function(progress, message) for updates
        
        Returns:
            Path to the generated searchable PDF
        """
        if not image_paths:
            raise ValueError("No images provided")
        
        logger.info(f"Generating searchable PDF from {len(image_paths)} images")
        
        # Step 1: Convert images to PDF
        if progress_callback:
            progress_callback(0.1, "Converting images to PDF...")
        
        temp_pdf = self._images_to_pdf(image_paths, title)
        logger.info(f"Created initial PDF: {temp_pdf}")
        
        # Step 2: Run OCR to add text layer
        if progress_callback:
            progress_callback(0.4, "Running OCR to make PDF searchable...")
        
        self._add_ocr_layer(temp_pdf, output_path)
        
        # Clean up temp file
        if os.path.exists(temp_pdf):
            os.unlink(temp_pdf)
        
        if progress_callback:
            progress_callback(1.0, "Searchable PDF generated!")
        
        logger.info(f"Searchable PDF created: {output_path}")
        return output_path
    
    def _images_to_pdf(self, image_paths: List[str], title: Optional[str] = None) -> str:
        """Convert images to a PDF file"""
        # Create temp file for initial PDF
        fd, temp_pdf = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        
        # Prepare images - ensure they're valid
        valid_images = []
        for path in image_paths:
            try:
                # Verify image can be opened
                with Image.open(path) as img:
                    # Convert to RGB if needed (removes alpha channel)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = img.convert('RGB')
                        # Save to temp file
                        temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        rgb_img.save(temp_img.name, 'JPEG', quality=95)
                        valid_images.append(temp_img.name)
                    else:
                        valid_images.append(path)
            except Exception as e:
                logger.warning(f"Skipping invalid image {path}: {e}")
        
        if not valid_images:
            raise ValueError("No valid images to convert")
        
        # Convert to PDF using img2pdf
        with open(temp_pdf, "wb") as f:
            f.write(img2pdf.convert(valid_images))
        
        return temp_pdf
    
    def _add_ocr_layer(self, input_pdf: str, output_pdf: str):
        """Add OCR text layer to PDF using ocrmypdf"""
        try:
            # Build ocrmypdf command
            cmd = [
                "ocrmypdf",
                "--language", self.language,
                "--output-type", "pdf",
                "--optimize", str(self.optimize),
                "--skip-text",  # Don't re-OCR pages that already have text
                "--force-ocr",  # Force OCR even if text detected
            ]
            
            if self.deskew:
                cmd.append("--deskew")
            
            if self.clean:
                cmd.append("--clean")
            
            # Add input and output
            cmd.extend([input_pdf, output_pdf])
            
            logger.info(f"Running OCR: {' '.join(cmd)}")
            
            # Run ocrmypdf
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"ocrmypdf error: {result.stderr}")
                # Fall back to just copying the PDF without OCR
                import shutil
                shutil.copy(input_pdf, output_pdf)
                logger.warning("Using PDF without OCR layer due to error")
            else:
                logger.info("OCR completed successfully")
                
        except FileNotFoundError:
            logger.error("ocrmypdf not found. Install with: brew install ocrmypdf")
            # Fall back to PDF without OCR
            import shutil
            shutil.copy(input_pdf, output_pdf)
        except subprocess.TimeoutExpired:
            logger.error("OCR timed out")
            import shutil
            shutil.copy(input_pdf, output_pdf)


def generate_searchable_pdf(
    image_paths: List[str],
    output_path: str,
    title: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> str:
    """
    Convenience function to generate a searchable PDF
    
    Args:
        image_paths: List of image file paths
        output_path: Output PDF path
        title: PDF title
        progress_callback: Progress updates
    
    Returns:
        Path to generated PDF
    """
    generator = SearchablePDFGenerator()
    return generator.generate(image_paths, output_path, title, progress_callback)
