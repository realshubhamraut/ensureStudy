"""
Document Processing Tasks for Celery Worker
Handles OCR, chunking, embedding, and indexing of teacher materials.
"""
import os
import json
import time
import logging
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime

from . import celery_app

logger = logging.getLogger(__name__)


# ============================================================
# TASK: Main Document Processing Orchestrator
# ============================================================

@celery_app.task(bind=True, max_retries=3)
def process_document(self, doc_id: str) -> Dict[str, Any]:
    """
    Main orchestrator task for document processing.
    Coordinates: fetch → extract pages → OCR → chunk → embed → index
    
    Args:
        doc_id: Document ID to process
        
    Returns:
        Processing result with status and metrics
    """
    start_time = time.time()
    logger.info(f"[INGEST][doc:{doc_id}] Starting document processing")
    
    try:
        # Import services (lazy to avoid circular imports)
        from ..services.s3_storage import get_storage_service
        from ..services.document_processor import DocumentProcessor
        
        storage = get_storage_service()
        processor = DocumentProcessor()
        
        # Update status to processing
        _update_document_status(doc_id, 'processing')
        
        # Step 1: Fetch document metadata and file
        logger.info(f"[INGEST][doc:{doc_id}] Fetching document from storage")
        doc_info = _get_document_info(doc_id)
        if not doc_info:
            raise ValueError(f"Document {doc_id} not found")
        
        file_content = storage.download_file(doc_info['s3_path'])
        if not file_content:
            raise ValueError(f"Failed to download file: {doc_info['s3_path']}")
        
        # Step 2: Check idempotency (file hash)
        file_hash = hashlib.sha256(file_content).hexdigest()
        if file_hash != doc_info.get('file_hash'):
            logger.warning(f"[INGEST][doc:{doc_id}] File hash mismatch")
        
        # Step 3: Extract pages from PDF
        logger.info(f"[INGEST][doc:{doc_id}] Extracting pages")
        pages_result = processor.extract_pages(file_content, doc_info['mime_type'])
        
        if not pages_result['success']:
            raise ValueError(f"Page extraction failed: {pages_result.get('error')}")
        
        page_count = pages_result['page_count']
        logger.info(f"[INGEST][doc:{doc_id}] Extracted {page_count} pages")
        
        # Step 4: OCR each page
        ocr_results = []
        for page_num, page_image in enumerate(pages_result['page_images'], start=1):
            logger.info(f"[INGEST][doc:{doc_id}][worker:ocr] page={page_num}")
            ocr_result = processor.ocr_page(page_image, page_num)
            ocr_results.append(ocr_result)
            
            # Save page JSON to S3
            page_json = json.dumps(ocr_result, ensure_ascii=False)
            storage.upload_processed_json(page_json, doc_id, page_num)
            
            # Update database
            _save_page_result(doc_id, page_num, ocr_result)
            
            logger.info(
                f"[INGEST][doc:{doc_id}][worker:ocr] page={page_num} "
                f"method={ocr_result.get('method')} "
                f"blocks={ocr_result.get('block_count', 0)} "
                f"avg_conf={ocr_result.get('avg_confidence', 0):.2f}"
            )
        
        # Step 5: Chunk and embed
        logger.info(f"[INGEST][doc:{doc_id}][worker:chunk] Creating chunks")
        chunks_result = processor.chunk_and_embed(doc_id, ocr_results, doc_info)
        
        logger.info(
            f"[INGEST][doc:{doc_id}][worker:chunk] "
            f"created_chunks={chunks_result['chunk_count']} "
            f"total_tokens={chunks_result['total_tokens']}"
        )
        
        # Step 6: Index to Qdrant
        logger.info(f"[INGEST][doc:{doc_id}][qdrant] Indexing vectors")
        index_result = processor.index_to_qdrant(
            doc_id=doc_id,
            class_id=doc_info['class_id'],
            chunks=chunks_result['chunks']
        )
        
        logger.info(
            f"[INGEST][doc:{doc_id}][qdrant] "
            f"inserted_vectors={index_result['indexed_count']} "
            f"collection=class_{doc_info['class_id']}_docs"
        )
        
        # Step 7: Create quality report
        quality_report = _create_quality_report(
            doc_id=doc_id,
            ocr_results=ocr_results,
            chunks_result=chunks_result,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        # Check if manual review needed
        requires_review = quality_report['avg_ocr_confidence'] < 0.7
        
        # Update final status
        _update_document_status(
            doc_id,
            'indexed',
            requires_review=requires_review
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[INGEST][doc:{doc_id}] status=indexed "
            f"indexed_count={index_result['indexed_count']} "
            f"avg_ocr_confidence={quality_report['avg_ocr_confidence']:.2f} "
            f"processing_time={elapsed_ms}ms"
        )
        
        return {
            'success': True,
            'doc_id': doc_id,
            'status': 'indexed',
            'page_count': page_count,
            'chunk_count': chunks_result['chunk_count'],
            'indexed_count': index_result['indexed_count'],
            'avg_ocr_confidence': quality_report['avg_ocr_confidence'],
            'requires_review': requires_review,
            'processing_time_ms': elapsed_ms
        }
        
    except Exception as e:
        logger.error(f"[INGEST][doc:{doc_id}] Error: {e}", exc_info=True)
        
        # Update status to error
        _update_document_status(doc_id, 'error', error_message=str(e))
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        
        return {
            'success': False,
            'doc_id': doc_id,
            'status': 'error',
            'error': str(e)
        }


# ============================================================
# Helper Functions
# ============================================================

def _get_document_info(doc_id: str) -> Optional[Dict[str, Any]]:
    """Get document info from database."""
    try:
        import httpx
        
        core_service_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        response = httpx.get(
            f"{core_service_url}/api/internal/documents/{doc_id}",
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        return None


def _update_document_status(
    doc_id: str,
    status: str,
    requires_review: bool = False,
    error_message: Optional[str] = None
):
    """Update document status in database."""
    try:
        import httpx
        
        core_service_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        httpx.patch(
            f"{core_service_url}/api/internal/documents/{doc_id}/status",
            json={
                'status': status,
                'requires_manual_review': requires_review,
                'error_message': error_message
            },
            timeout=10.0
        )
        
    except Exception as e:
        logger.error(f"Failed to update document status: {e}")


def _save_page_result(doc_id: str, page_number: int, ocr_result: Dict[str, Any]):
    """Save page OCR result to database."""
    try:
        import httpx
        
        core_service_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        httpx.post(
            f"{core_service_url}/api/internal/documents/{doc_id}/pages",
            json={
                'page_number': page_number,
                'ocr_confidence': ocr_result.get('avg_confidence'),
                'text_length': ocr_result.get('text_length', 0),
                'block_count': ocr_result.get('block_count', 0),
                'ocr_method': ocr_result.get('method')
            },
            timeout=10.0
        )
        
    except Exception as e:
        logger.error(f"Failed to save page result: {e}")


def _create_quality_report(
    doc_id: str,
    ocr_results: List[Dict[str, Any]],
    chunks_result: Dict[str, Any],
    processing_time_ms: int
) -> Dict[str, Any]:
    """Create and save quality report for document."""
    
    confidences = [r.get('avg_confidence', 0) for r in ocr_results if r.get('avg_confidence')]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    min_confidence = min(confidences) if confidences else 0
    
    flagged_pages = [
        r.get('page_number', i+1)
        for i, r in enumerate(ocr_results)
        if r.get('avg_confidence', 1) < 0.7
    ]
    
    report = {
        'document_id': doc_id,
        'avg_ocr_confidence': avg_confidence,
        'min_ocr_confidence': min_confidence,
        'pages_processed': len(ocr_results),
        'pages_failed': len([r for r in ocr_results if not r.get('success', True)]),
        'flagged_pages': flagged_pages,
        'total_chunks': chunks_result.get('chunk_count', 0),
        'total_tokens': chunks_result.get('total_tokens', 0),
        'processing_time_ms': processing_time_ms
    }
    
    # Save to database
    try:
        import httpx
        
        core_service_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
        httpx.post(
            f"{core_service_url}/api/internal/documents/{doc_id}/quality-report",
            json=report,
            timeout=10.0
        )
        
    except Exception as e:
        logger.error(f"Failed to save quality report: {e}")
    
    return report


# ============================================================
# Individual Tasks (for distributed processing)
# ============================================================

@celery_app.task(bind=True, max_retries=3)
def extract_pages(self, doc_id: str, s3_path: str, mime_type: str) -> Dict[str, Any]:
    """Extract pages from PDF document."""
    try:
        from ..services.s3_storage import get_storage_service
        from ..services.document_processor import DocumentProcessor
        
        storage = get_storage_service()
        processor = DocumentProcessor()
        
        file_content = storage.download_file(s3_path)
        if not file_content:
            raise ValueError(f"Failed to download file: {s3_path}")
        
        result = processor.extract_pages(file_content, mime_type)
        
        # Store page images in S3
        page_paths = []
        for i, page_image in enumerate(result.get('page_images', []), start=1):
            # page_image is PIL Image or bytes
            # Save to S3 as PNG
            pass  # Implementation depends on image format
        
        return {
            'success': result['success'],
            'page_count': result['page_count'],
            'page_paths': page_paths
        }
        
    except Exception as e:
        logger.error(f"[extract_pages] Error: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def ocr_page(self, doc_id: str, page_number: int, page_image_path: str) -> Dict[str, Any]:
    """Run OCR on a single page."""
    try:
        from ..services.s3_storage import get_storage_service
        from ..services.document_processor import DocumentProcessor
        
        storage = get_storage_service()
        processor = DocumentProcessor()
        
        # Download page image
        page_image = storage.download_file(page_image_path)
        if not page_image:
            raise ValueError(f"Failed to download page image: {page_image_path}")
        
        # Run OCR
        result = processor.ocr_page(page_image, page_number)
        
        # Save result to S3
        page_json = json.dumps(result, ensure_ascii=False)
        storage.upload_processed_json(page_json, doc_id, page_number)
        
        return result
        
    except Exception as e:
        logger.error(f"[ocr_page] Error: {e}")
        raise self.retry(exc=e, countdown=60)
