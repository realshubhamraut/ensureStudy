"""
Unit Tests for Document Processing Pipeline
Tests OCR, chunking, embedding, and indexing components.
"""
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image
import numpy as np


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (612, 792), color='white')
    return img


@pytest.fixture
def sample_pdf_content():
    """Return minimal valid PDF bytes."""
    # Minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF
"""
    return pdf_content


@pytest.fixture
def sample_ocr_result():
    """Sample OCR result for testing."""
    return {
        'page_number': 1,
        'blocks': [
            {
                'text': 'This is a test document',
                'bbox': [50, 100, 400, 130],
                'confidence': 0.95,
                'block_type': 'text'
            },
            {
                'text': 'Contact email@test.com for more info',
                'bbox': [50, 150, 400, 180],
                'confidence': 0.88,
                'block_type': 'text'
            }
        ],
        'full_text': 'This is a test document. Contact email@test.com for more info',
        'avg_confidence': 0.915,
        'block_count': 2,
        'text_length': 62,
        'method': 'tesseract',
        'success': True
    }


# ============================================================
# Document Processor Tests
# ============================================================

class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    def test_pii_redaction_email(self):
        """Test that email addresses are redacted."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        text = "Contact john.doe@example.com for assistance"
        redacted = processor._redact_pii(text)
        
        assert "[EMAIL_REDACTED]" in redacted
        assert "john.doe@example.com" not in redacted
    
    def test_pii_redaction_phone(self):
        """Test that phone numbers are redacted."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        text = "Call us at +91-9876543210 or 123-456-7890"
        redacted = processor._redact_pii(text)
        
        assert "[PHONE_REDACTED]" in redacted
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Create a long text
        text = " ".join(["word"] * 1000)
        
        chunks = processor._chunk_text(
            text=text,
            page_number=1,
            blocks=[],
            doc_id="test-doc-123",
            chunk_size=100,
            overlap=20
        )
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert 'id' in chunk
            assert 'document_id' in chunk
            assert 'page_number' in chunk
            assert 'text' in chunk
            assert 'token_count' in chunk
    
    def test_chunk_text_with_overlap(self):
        """Test that chunks have proper overlap."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Create text with numbered words for easy verification
        text = " ".join([f"word{i}" for i in range(200)])
        
        chunks = processor._chunk_text(
            text=text,
            page_number=1,
            blocks=[],
            doc_id="test-doc-123",
            chunk_size=50,
            overlap=10
        )
        
        # Check overlap exists between consecutive chunks
        if len(chunks) >= 2:
            chunk1_words = set(chunks[0]['text'].split())
            chunk2_words = set(chunks[1]['text'].split())
            overlap_words = chunk1_words.intersection(chunk2_words)
            assert len(overlap_words) > 0


# ============================================================
# S3 Storage Tests
# ============================================================

class TestS3Storage:
    """Tests for S3StorageService."""
    
    def test_compute_file_hash(self):
        """Test file hash computation."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.s3_storage import S3StorageService
        
        storage = S3StorageService()
        
        content = b"Hello, World!"
        hash1 = storage.compute_file_hash(content)
        hash2 = storage.compute_file_hash(content)
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_compute_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.s3_storage import S3StorageService
        
        storage = S3StorageService()
        
        hash1 = storage.compute_file_hash(b"Content A")
        hash2 = storage.compute_file_hash(b"Content B")
        
        assert hash1 != hash2
    
    @patch('app.services.s3_storage.Minio')
    def test_upload_local_fallback(self, mock_minio):
        """Test local fallback when MinIO unavailable."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from app.services.s3_storage import S3StorageService
        
        # Force local fallback
        mock_minio.side_effect = Exception("Connection failed")
        
        storage = S3StorageService()
        storage.client = None  # Simulate no client
        
        result = storage._upload_local(
            file_content=b"Test content",
            s3_path="test/path/file.txt",
            file_hash="abc123",
            file_size=12
        )
        
        assert result.success is True
        assert result.s3_path.startswith("file://")


# ============================================================
# Document Tasks Tests
# ============================================================

class TestDocumentTasks:
    """Tests for Celery document tasks."""
    
    @patch('app.workers.document_tasks._get_document_info')
    @patch('app.workers.document_tasks._update_document_status')
    def test_process_document_not_found(self, mock_update, mock_get_info):
        """Test handling of non-existent document."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        mock_get_info.return_value = None
        mock_update.return_value = None
        
        from app.workers.document_tasks import process_document
        
        # Should handle gracefully
        result = process_document.apply(args=["non-existent-doc"]).get()
        
        assert result['success'] is False
        assert 'error' in result


# ============================================================
# API Endpoint Tests
# ============================================================

class TestDocumentAPI:
    """Tests for document API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        import sys
        sys.path.insert(0, 'backend/ai-service')
        
        from fastapi.testclient import TestClient
        from app.main import app
        
        return TestClient(app)
    
    def test_upload_invalid_file_type(self, client):
        """Test rejection of invalid file types."""
        # Create a text file (not allowed)
        files = {
            'file': ('test.txt', b'Hello World', 'text/plain')
        }
        
        response = client.post(
            '/api/classrooms/test-class/materials/upload',
            files=files
        )
        
        assert response.status_code == 400
        assert 'Unsupported file type' in response.json().get('detail', '')
    
    def test_sidebar_not_found(self, client):
        """Test sidebar API with non-existent document."""
        response = client.get(
            '/api/ai-tutor/documents/non-existent-doc/sidebar'
        )
        
        # May return 404 or 500 depending on implementation
        assert response.status_code in [404, 500]


# ============================================================
# Integration Tests
# ============================================================

@pytest.mark.integration
class TestDocumentPipelineIntegration:
    """Integration tests for full document pipeline."""
    
    @pytest.mark.skip(reason="Requires running services")
    def test_full_pipeline(self, sample_pdf_content):
        """Test complete upload → process → index pipeline."""
        import requests
        
        AI_SERVICE_URL = "http://localhost:8001"
        
        # 1. Upload document
        files = {
            'file': ('test.pdf', sample_pdf_content, 'application/pdf')
        }
        
        response = requests.post(
            f"{AI_SERVICE_URL}/api/classrooms/test-class/materials/upload",
            files=files
        )
        
        assert response.status_code == 200
        data = response.json()
        doc_id = data['doc_id']
        
        # 2. Check status (may need polling)
        import time
        max_attempts = 10
        for _ in range(max_attempts):
            status_response = requests.get(
                f"{AI_SERVICE_URL}/api/classrooms/test-class/materials/{doc_id}/status"
            )
            status = status_response.json()['status']
            
            if status in ['indexed', 'error']:
                break
            
            time.sleep(2)
        
        assert status == 'indexed'
        
        # 3. Query sidebar
        sidebar_response = requests.get(
            f"{AI_SERVICE_URL}/api/ai-tutor/documents/{doc_id}/sidebar?query=test"
        )
        
        assert sidebar_response.status_code == 200
        sidebar_data = sidebar_response.json()
        assert 'top_matches' in sidebar_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
