"""
S3 Storage Service for Document Management
Provides S3-compatible storage operations using MinIO for local dev
and AWS S3 for production.
"""
import os
import hashlib
import logging
from typing import Optional, BinaryIO, Tuple
from datetime import timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of file upload operation"""
    success: bool
    s3_path: str
    file_hash: str
    file_size: int
    error: Optional[str] = None


class S3StorageService:
    """
    S3-compatible storage service for document files.
    Uses MinIO for local development, AWS S3 for production.
    """
    
    def __init__(self):
        self.endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.access_key = os.getenv('MINIO_ACCESS_KEY', 'ensurestudy')
        self.secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        self.bucket = os.getenv('MINIO_BUCKET', 'ensurestudy-documents')
        self.use_ssl = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'
        
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize MinIO/S3 client."""
        try:
            from minio import Minio
            from minio.error import S3Error
            
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.use_ssl
            )
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"[S3] Created bucket: {self.bucket}")
            
            logger.info(f"[S3] Connected to {self.endpoint}")
            
        except ImportError:
            logger.warning("[S3] minio package not installed. Using local filesystem.")
            self.client = None
        except Exception as e:
            logger.error(f"[S3] Failed to connect: {e}")
            self.client = None
    
    def compute_file_hash(self, file_content: bytes) -> str:
        """Compute SHA256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def upload_file(
        self,
        file_content: bytes,
        class_id: str,
        doc_id: str,
        filename: str,
        content_type: str = 'application/octet-stream'
    ) -> UploadResult:
        """
        Upload a file to S3 storage.
        
        Args:
            file_content: Raw file bytes
            class_id: Classroom ID
            doc_id: Document ID
            filename: Original filename
            content_type: MIME type
            
        Returns:
            UploadResult with S3 path and file hash
        """
        file_hash = self.compute_file_hash(file_content)
        file_size = len(file_content)
        s3_path = f"classrooms/{class_id}/materials/{doc_id}/{filename}"
        
        if not self.client:
            # Fallback to local filesystem
            return self._upload_local(file_content, s3_path, file_hash, file_size)
        
        try:
            from io import BytesIO
            
            self.client.put_object(
                self.bucket,
                s3_path,
                BytesIO(file_content),
                length=file_size,
                content_type=content_type
            )
            
            logger.info(f"[S3] Uploaded: s3://{self.bucket}/{s3_path} ({file_size} bytes)")
            
            return UploadResult(
                success=True,
                s3_path=f"s3://{self.bucket}/{s3_path}",
                file_hash=file_hash,
                file_size=file_size
            )
            
        except Exception as e:
            logger.error(f"[S3] Upload failed: {e}")
            return UploadResult(
                success=False,
                s3_path="",
                file_hash=file_hash,
                file_size=file_size,
                error=str(e)
            )
    
    def _upload_local(
        self,
        file_content: bytes,
        s3_path: str,
        file_hash: str,
        file_size: int
    ) -> UploadResult:
        """Fallback: store file locally."""
        local_dir = os.path.join(os.getcwd(), 'data', 'storage')
        full_path = os.path.join(local_dir, s3_path)
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"[S3][LOCAL] Saved to: {full_path}")
        
        return UploadResult(
            success=True,
            s3_path=f"file://{full_path}",
            file_hash=file_hash,
            file_size=file_size
        )
    
    def download_file(self, s3_path: str) -> Optional[bytes]:
        """
        Download file from S3 storage.
        
        Args:
            s3_path: Full S3 path (s3://bucket/key or file://path)
            
        Returns:
            File content as bytes, or None if not found
        """
        if s3_path.startswith("file://"):
            # Local filesystem
            local_path = s3_path.replace("file://", "")
            if os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    return f.read()
            return None
        
        if not self.client:
            return None
        
        try:
            # Parse s3://bucket/key format
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ""
            
            response = self.client.get_object(bucket, key)
            content = response.read()
            response.close()
            response.release_conn()
            
            return content
            
        except Exception as e:
            logger.error(f"[S3] Download failed: {e}")
            return None
    
    def get_presigned_url(
        self,
        s3_path: str,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.
        
        Args:
            s3_path: Full S3 path
            expires: URL expiration time (default 1 hour)
            
        Returns:
            Presigned URL string, or None on error
        """
        if s3_path.startswith("file://"):
            # For local files, return the path directly
            return s3_path.replace("file://", "")
        
        if not self.client:
            return None
        
        try:
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ""
            
            url = self.client.presigned_get_object(
                bucket,
                key,
                expires=expires
            )
            
            return url
            
        except Exception as e:
            logger.error(f"[S3] Presigned URL failed: {e}")
            return None
    
    def upload_processed_json(
        self,
        json_content: str,
        doc_id: str,
        page_number: int
    ) -> UploadResult:
        """
        Upload processed page JSON to S3.
        
        Args:
            json_content: JSON string
            doc_id: Document ID
            page_number: Page number
            
        Returns:
            UploadResult with S3 path
        """
        s3_path = f"processed/{doc_id}/pages/{page_number}.json"
        content_bytes = json_content.encode('utf-8')
        
        if not self.client:
            return self._upload_local(
                content_bytes,
                s3_path,
                self.compute_file_hash(content_bytes),
                len(content_bytes)
            )
        
        try:
            from io import BytesIO
            
            self.client.put_object(
                self.bucket,
                s3_path,
                BytesIO(content_bytes),
                length=len(content_bytes),
                content_type='application/json'
            )
            
            return UploadResult(
                success=True,
                s3_path=f"s3://{self.bucket}/{s3_path}",
                file_hash=self.compute_file_hash(content_bytes),
                file_size=len(content_bytes)
            )
            
        except Exception as e:
            logger.error(f"[S3] Upload processed JSON failed: {e}")
            return UploadResult(
                success=False,
                s3_path="",
                file_hash="",
                file_size=0,
                error=str(e)
            )
    
    def delete_file(self, s3_path: str) -> bool:
        """Delete a file from S3 storage."""
        if s3_path.startswith("file://"):
            local_path = s3_path.replace("file://", "")
            try:
                os.remove(local_path)
                return True
            except:
                return False
        
        if not self.client:
            return False
        
        try:
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ""
            
            self.client.remove_object(bucket, key)
            return True
            
        except Exception as e:
            logger.error(f"[S3] Delete failed: {e}")
            return False


# Singleton instance
_storage_service: Optional[S3StorageService] = None


def get_storage_service() -> S3StorageService:
    """Get or create S3 storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = S3StorageService()
    return _storage_service
