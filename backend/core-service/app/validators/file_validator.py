"""
File Validator - Secure file upload validation

Validates:
- File type (extension and MIME type)
- File size
- Filename sanitization
- Basic malware patterns
"""
import os
import re
import hashlib
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ALLOWED_EXTENSIONS = {
    # Documents
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".txt": "text/plain",
    
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    
    # Videos
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}

# Max file sizes (in bytes)
MAX_FILE_SIZES = {
    "document": 50 * 1024 * 1024,  # 50MB for documents
    "image": 10 * 1024 * 1024,     # 10MB for images
    "video": 200 * 1024 * 1024,    # 200MB for videos
    "default": 50 * 1024 * 1024,   # 50MB default
}

# Dangerous patterns in filenames
DANGEROUS_PATTERNS = [
    r"\.\.",         # Directory traversal
    r"^/",           # Absolute path
    r"\\",           # Windows path separator
    r"\x00",         # Null byte
    r"[<>:\"|?*]",   # Windows reserved chars
]


# ============================================================================
# File Validator
# ============================================================================

@dataclass
class ValidationResult:
    """Result of file validation"""
    valid: bool
    error: Optional[str] = None
    sanitized_filename: Optional[str] = None
    file_type: Optional[str] = None
    file_hash: Optional[str] = None


class FileValidator:
    """
    Secure file upload validation.
    
    Usage:
        validator = FileValidator()
        result = validator.validate(file, filename)
        
        if not result.valid:
            return {"error": result.error}, 400
        
        # Use result.sanitized_filename for storage
    """
    
    def __init__(
        self,
        allowed_extensions: dict = None,
        max_file_sizes: dict = None
    ):
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS
        self.max_file_sizes = max_file_sizes or MAX_FILE_SIZES
    
    def validate(
        self,
        file_content: bytes,
        filename: str,
        expected_type: str = "document"
    ) -> ValidationResult:
        """
        Validate uploaded file.
        
        Args:
            file_content: File bytes
            filename: Original filename
            expected_type: document, image, video
            
        Returns:
            ValidationResult with validation status
        """
        # Validate filename
        sanitized = self.sanitize_filename(filename)
        if not sanitized:
            return ValidationResult(
                valid=False,
                error="Invalid filename"
            )
        
        # Validate extension
        ext = os.path.splitext(sanitized)[1].lower()
        if ext not in self.allowed_extensions:
            return ValidationResult(
                valid=False,
                error=f"File type not allowed: {ext}"
            )
        
        # Validate file size
        max_size = self.max_file_sizes.get(expected_type, self.max_file_sizes["default"])
        if len(file_content) > max_size:
            max_mb = max_size / (1024 * 1024)
            return ValidationResult(
                valid=False,
                error=f"File too large. Maximum size: {max_mb:.0f}MB"
            )
        
        # Validate MIME type
        detected_mime = self._detect_mime_type(file_content)
        expected_mime = self.allowed_extensions.get(ext)
        
        if detected_mime and expected_mime:
            if not self._mime_matches(detected_mime, expected_mime):
                return ValidationResult(
                    valid=False,
                    error=f"File content does not match extension"
                )
        
        # Check for dangerous content
        if self._has_dangerous_content(file_content):
            return ValidationResult(
                valid=False,
                error="File contains potentially dangerous content"
            )
        
        # Calculate hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Determine file type category
        file_type = self._get_file_type(ext)
        
        return ValidationResult(
            valid=True,
            sanitized_filename=sanitized,
            file_type=file_type,
            file_hash=file_hash
        )
    
    def sanitize_filename(self, filename: str) -> Optional[str]:
        """Sanitize filename to prevent security issues"""
        if not filename:
            return None
        
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                logger.warning(f"[FileValidator] Dangerous pattern in filename: {filename}")
                return None
        
        # Remove leading/trailing whitespace
        filename = filename.strip()
        
        # Remove any path components
        filename = os.path.basename(filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        
        # Remove non-ASCII characters (optional: keep Unicode letters)
        filename = re.sub(r'[^\w\-_.]', '', filename)
        
        if not filename or filename in [".", ".."]:
            return None
        
        return filename
    
    def validate_file_type(self, filename: str) -> Tuple[bool, Optional[str]]:
        """Quick check if file type is allowed"""
        ext = os.path.splitext(filename)[1].lower()
        if ext in self.allowed_extensions:
            return True, None
        return False, f"File type not allowed: {ext}"
    
    def validate_file_size(self, size_bytes: int, file_type: str = "default") -> Tuple[bool, Optional[str]]:
        """Quick check if file size is allowed"""
        max_size = self.max_file_sizes.get(file_type, self.max_file_sizes["default"])
        if size_bytes <= max_size:
            return True, None
        max_mb = max_size / (1024 * 1024)
        return False, f"File too large. Maximum: {max_mb:.0f}MB"
    
    def _detect_mime_type(self, content: bytes) -> Optional[str]:
        """Detect MIME type from file content"""
        try:
            import magic
            mime = magic.from_buffer(content, mime=True)
            return mime
        except ImportError:
            # Fallback: check magic bytes for common types
            return self._detect_by_magic_bytes(content)
        except Exception:
            return None
    
    def _detect_by_magic_bytes(self, content: bytes) -> Optional[str]:
        """Detect file type by magic bytes"""
        if len(content) < 8:
            return None
        
        # PDF
        if content[:4] == b'%PDF':
            return "application/pdf"
        
        # PNG
        if content[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        
        # JPEG
        if content[:2] == b'\xff\xd8':
            return "image/jpeg"
        
        # GIF
        if content[:6] in [b'GIF87a', b'GIF89a']:
            return "image/gif"
        
        # ZIP-based (docx, pptx, xlsx)
        if content[:4] == b'PK\x03\x04':
            return "application/zip"  # Generic, could be docx/pptx
        
        return None
    
    def _mime_matches(self, detected: str, expected: str) -> bool:
        """Check if MIME types match (with some flexibility)"""
        if detected == expected:
            return True
        
        # ZIP-based documents
        if detected == "application/zip" and "officedocument" in expected:
            return True
        
        # Image subtypes
        if detected.startswith("image/") and expected.startswith("image/"):
            return True  # Be lenient with image subtypes
        
        return False
    
    def _has_dangerous_content(self, content: bytes) -> bool:
        """Check for potentially dangerous content"""
        # Check for common script patterns (in text-like files)
        dangerous_patterns = [
            b'<script',
            b'javascript:',
            b'data:text/html',
            b'<?php',
            b'<%',  # ASP
        ]
        
        # Only check first 8KB (for performance)
        check_content = content[:8192].lower()
        
        for pattern in dangerous_patterns:
            if pattern in check_content:
                logger.warning(f"[FileValidator] Dangerous content pattern detected")
                return True
        
        return False
    
    def _get_file_type(self, extension: str) -> str:
        """Categorize file by extension"""
        ext = extension.lower()
        
        if ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt']:
            return "document"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return "image"
        elif ext in ['.mp4', '.webm', '.mov']:
            return "video"
        else:
            return "other"


# ============================================================================
# Singleton
# ============================================================================

_file_validator: Optional[FileValidator] = None


def get_file_validator() -> FileValidator:
    """Get or create file validator singleton"""
    global _file_validator
    if _file_validator is None:
        _file_validator = FileValidator()
    return _file_validator
