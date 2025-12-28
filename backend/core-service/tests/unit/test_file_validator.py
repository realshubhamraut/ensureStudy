"""
Unit Tests for File Validator Service
"""
import pytest
from app.validators.file_validator import (
    FileValidator, ValidationResult, get_file_validator,
    ALLOWED_EXTENSIONS, MAX_FILE_SIZES
)


class TestFileValidator:
    """Tests for FileValidator class"""
    
    @pytest.fixture
    def validator(self):
        return FileValidator()
    
    # =========================================================================
    # File Type Validation Tests
    # =========================================================================
    
    def test_validate_file_type_pdf(self, validator):
        """Test PDF file type validation"""
        valid, error = validator.validate_file_type("document.pdf")
        assert valid is True
        assert error is None
    
    def test_validate_file_type_pptx(self, validator):
        """Test PPTX file type validation"""
        valid, error = validator.validate_file_type("slides.pptx")
        assert valid is True
    
    def test_validate_file_type_images(self, validator):
        """Test image file types"""
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            valid, error = validator.validate_file_type(f"image{ext}")
            assert valid is True, f"Failed for {ext}"
    
    def test_validate_file_type_invalid_exe(self, validator):
        """Test that .exe files are rejected"""
        valid, error = validator.validate_file_type("malware.exe")
        assert valid is False
        assert "not allowed" in error.lower()
    
    def test_validate_file_type_invalid_sh(self, validator):
        """Test that shell scripts are rejected"""
        valid, error = validator.validate_file_type("script.sh")
        assert valid is False
    
    def test_validate_file_type_invalid_js(self, validator):
        """Test that JavaScript files are rejected"""
        valid, error = validator.validate_file_type("code.js")
        assert valid is False
    
    # =========================================================================
    # File Size Validation Tests
    # =========================================================================
    
    def test_validate_file_size_within_limit(self, validator):
        """Test file within size limit"""
        size = 10 * 1024 * 1024  # 10MB
        valid, error = validator.validate_file_size(size, "document")
        assert valid is True
        assert error is None
    
    def test_validate_file_size_at_limit(self, validator):
        """Test file at exact limit"""
        size = 50 * 1024 * 1024  # 50MB (document limit)
        valid, error = validator.validate_file_size(size, "document")
        assert valid is True
    
    def test_validate_file_size_exceeds_limit(self, validator):
        """Test file exceeding size limit"""
        size = 100 * 1024 * 1024  # 100MB
        valid, error = validator.validate_file_size(size, "document")
        assert valid is False
        assert "too large" in error.lower()
    
    def test_validate_file_size_image_limit(self, validator):
        """Test image size limit (10MB)"""
        valid, error = validator.validate_file_size(15 * 1024 * 1024, "image")
        assert valid is False
    
    # =========================================================================
    # Filename Sanitization Tests
    # =========================================================================
    
    def test_sanitize_filename_normal(self, validator):
        """Test normal filename sanitization"""
        result = validator.sanitize_filename("document.pdf")
        assert result == "document.pdf"
    
    def test_sanitize_filename_spaces(self, validator):
        """Test filename with spaces"""
        result = validator.sanitize_filename("my document.pdf")
        assert result == "my_document.pdf"
    
    def test_sanitize_filename_path_traversal(self, validator):
        """Test path traversal attack prevention"""
        result = validator.sanitize_filename("../../etc/passwd")
        assert result is None  # Dangerous, rejected
    
    def test_sanitize_filename_absolute_path(self, validator):
        """Test absolute path rejection"""
        result = validator.sanitize_filename("/etc/passwd")
        assert result is None
    
    def test_sanitize_filename_windows_path(self, validator):
        """Test Windows path separator rejection"""
        result = validator.sanitize_filename("..\\..\\system32\\config")
        assert result is None
    
    def test_sanitize_filename_null_byte(self, validator):
        """Test null byte injection prevention"""
        result = validator.sanitize_filename("file.pdf\x00.exe")
        assert result is None
    
    def test_sanitize_filename_special_chars(self, validator):
        """Test special character removal"""
        result = validator.sanitize_filename("file<>:\"|?*.pdf")
        assert result is None or "<" not in result
    
    def test_sanitize_filename_long_name(self, validator):
        """Test long filename truncation"""
        long_name = "a" * 300 + ".pdf"
        result = validator.sanitize_filename(long_name)
        assert result is not None
        assert len(result) <= 255
    
    def test_sanitize_filename_empty(self, validator):
        """Test empty filename rejection"""
        result = validator.sanitize_filename("")
        assert result is None
    
    def test_sanitize_filename_dot_only(self, validator):
        """Test dot-only filename rejection"""
        assert validator.sanitize_filename(".") is None
        assert validator.sanitize_filename("..") is None
    
    # =========================================================================
    # Full Validation Tests
    # =========================================================================
    
    def test_validate_valid_pdf(self, validator):
        """Test full validation with valid PDF"""
        # PDF magic bytes
        content = b'%PDF-1.4' + b'0' * 1000
        result = validator.validate(content, "document.pdf", "document")
        assert result.valid is True
        assert result.sanitized_filename == "document.pdf"
        assert result.file_type == "document"
        assert result.file_hash is not None
    
    def test_validate_valid_image(self, validator):
        """Test full validation with valid PNG"""
        # PNG magic bytes
        content = b'\x89PNG\r\n\x1a\n' + b'0' * 1000
        result = validator.validate(content, "image.png", "image")
        assert result.valid is True
        assert result.file_type == "image"
    
    def test_validate_dangerous_content(self, validator):
        """Test rejection of files with dangerous content"""
        content = b'<script>alert("xss")</script>'
        result = validator.validate(content, "readme.txt", "document")
        assert result.valid is False
        assert "dangerous" in result.error.lower()
    
    def test_validate_php_content(self, validator):
        """Test rejection of PHP code"""
        content = b'<?php echo shell_exec($_GET["cmd"]); ?>'
        result = validator.validate(content, "file.txt", "document")
        assert result.valid is False


class TestValidationResult:
    """Tests for ValidationResult dataclass"""
    
    def test_valid_result(self):
        result = ValidationResult(
            valid=True,
            sanitized_filename="file.pdf",
            file_type="document",
            file_hash="abc123"
        )
        assert result.valid is True
        assert result.error is None
    
    def test_invalid_result(self):
        result = ValidationResult(
            valid=False,
            error="File type not allowed"
        )
        assert result.valid is False
        assert result.error is not None


class TestFileValidatorSingleton:
    """Tests for singleton pattern"""
    
    def test_get_file_validator_singleton(self):
        v1 = get_file_validator()
        v2 = get_file_validator()
        assert v1 is v2
