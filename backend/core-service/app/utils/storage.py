"""
S3 Storage Service - AWS S3 integration for document storage

Supports:
- Upload files to S3
- Generate presigned URLs for direct upload/download
- Fallback to local storage for development
"""
import os
import uuid
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of file upload operation"""
    success: bool
    storage_url: str  # S3 URL or local path
    storage_type: str  # 's3' or 'local'
    file_key: str  # S3 key or filename
    size_bytes: int
    error: Optional[str] = None


class StorageService:
    """
    Unified storage service with S3 and local fallback.
    
    Environment variables:
    - AWS_ACCESS_KEY_ID: AWS access key
    - AWS_SECRET_ACCESS_KEY: AWS secret key
    - AWS_REGION: AWS region (default: us-east-1)
    - AWS_S3_BUCKET: S3 bucket name
    - STORAGE_MODE: 's3' or 'local' (default: auto-detect)
    - UPLOAD_FOLDER: Local storage folder (default: /tmp/notes_uploads)
    """
    
    def __init__(self):
        self.storage_mode = os.getenv("STORAGE_MODE", "auto")
        self.bucket_name = os.getenv("AWS_S3_BUCKET", "ensure-study-documents")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.local_folder = os.getenv("UPLOAD_FOLDER", "/tmp/notes_uploads")
        
        # Initialize S3 client if credentials available
        self.s3_client = None
        self._initialize_s3()
        
        # Determine effective storage mode
        if self.storage_mode == "auto":
            self.storage_mode = "s3" if self.s3_client else "local"
        
        logger.info(f"[Storage] Initialized with mode: {self.storage_mode}")
    
    def _initialize_s3(self):
        """Initialize S3 client if credentials are available"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            if access_key and secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=self.region
                )
                
                # Verify bucket exists (or create it)
                try:
                    self.s3_client.head_bucket(Bucket=self.bucket_name)
                    logger.info(f"[Storage] S3 bucket '{self.bucket_name}' verified")
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    if error_code == '404':
                        logger.warning(f"[Storage] Bucket '{self.bucket_name}' not found, creating...")
                        self._create_bucket()
                    else:
                        logger.warning(f"[Storage] S3 bucket check failed: {e}")
                        self.s3_client = None
            else:
                logger.info("[Storage] AWS credentials not found, using local storage")
                
        except ImportError:
            logger.warning("[Storage] boto3 not installed, using local storage")
        except Exception as e:
            logger.warning(f"[Storage] S3 initialization failed: {e}")
            self.s3_client = None
    
    def _create_bucket(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"[Storage] Created bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"[Storage] Failed to create bucket: {e}")
            self.s3_client = None
    
    def upload_file(
        self,
        file_data: bytes,
        filename: str,
        job_id: str,
        content_type: Optional[str] = None
    ) -> UploadResult:
        """
        Upload a file to storage (S3 or local).
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            job_id: Job ID for folder organization
            content_type: MIME type of file
            
        Returns:
            UploadResult with storage URL and metadata
        """
        file_key = f"notes/{job_id}/{filename}"
        size_bytes = len(file_data)
        
        if self.storage_mode == "s3" and self.s3_client:
            return self._upload_to_s3(file_data, file_key, content_type, size_bytes)
        else:
            return self._upload_to_local(file_data, file_key, size_bytes)
    
    def _upload_to_s3(
        self,
        file_data: bytes,
        file_key: str,
        content_type: Optional[str],
        size_bytes: int
    ) -> UploadResult:
        """Upload file to S3"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_key,
                Body=file_data,
                **extra_args
            )
            
            storage_url = f"s3://{self.bucket_name}/{file_key}"
            
            return UploadResult(
                success=True,
                storage_url=storage_url,
                storage_type="s3",
                file_key=file_key,
                size_bytes=size_bytes
            )
            
        except Exception as e:
            logger.error(f"[Storage] S3 upload failed: {e}")
            # Fallback to local
            return self._upload_to_local(file_data, file_key, size_bytes)
    
    def _upload_to_local(
        self,
        file_data: bytes,
        file_key: str,
        size_bytes: int
    ) -> UploadResult:
        """Upload file to local storage"""
        try:
            local_path = os.path.join(self.local_folder, file_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                f.write(file_data)
            
            return UploadResult(
                success=True,
                storage_url=local_path,
                storage_type="local",
                file_key=file_key,
                size_bytes=size_bytes
            )
            
        except Exception as e:
            logger.error(f"[Storage] Local upload failed: {e}")
            return UploadResult(
                success=False,
                storage_url="",
                storage_type="local",
                file_key=file_key,
                size_bytes=0,
                error=str(e)
            )
    
    def generate_presigned_upload_url(
        self,
        job_id: str,
        filename: str,
        content_type: str,
        expiration: int = 3600
    ) -> Optional[dict]:
        """
        Generate a presigned URL for direct browser upload to S3.
        
        Args:
            job_id: Job ID for folder organization
            filename: Target filename
            content_type: MIME type of file
            expiration: URL expiration in seconds
            
        Returns:
            dict with 'url' and 'fields' for form upload, or None if not available
        """
        if not self.s3_client:
            return None
        
        file_key = f"notes/{job_id}/{filename}"
        
        try:
            presigned = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=file_key,
                Fields={"Content-Type": content_type},
                Conditions=[
                    {"Content-Type": content_type},
                    ["content-length-range", 1, 500 * 1024 * 1024]  # 500MB max
                ],
                ExpiresIn=expiration
            )
            
            return {
                "url": presigned["url"],
                "fields": presigned["fields"],
                "file_key": file_key
            }
            
        except Exception as e:
            logger.error(f"[Storage] Failed to generate presigned URL: {e}")
            return None
    
    def generate_presigned_download_url(
        self,
        file_key: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for downloading a file from S3.
        
        Args:
            file_key: S3 object key
            expiration: URL expiration in seconds
            
        Returns:
            Presigned URL string or None if not available
        """
        if not self.s3_client:
            return None
        
        try:
            # Handle s3:// URLs
            if file_key.startswith("s3://"):
                file_key = file_key.split("/", 3)[3]  # Remove s3://bucket/
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_key
                },
                ExpiresIn=expiration
            )
            return url
            
        except Exception as e:
            logger.error(f"[Storage] Failed to generate download URL: {e}")
            return None
    
    def get_file_url(self, storage_url: str) -> str:
        """
        Get a URL for accessing a stored file.
        
        For S3: generates presigned download URL
        For local: returns the local path
        
        Args:
            storage_url: Storage URL from UploadResult
            
        Returns:
            Accessible URL or path
        """
        if storage_url.startswith("s3://"):
            presigned = self.generate_presigned_download_url(storage_url)
            return presigned or storage_url
        else:
            return storage_url
    
    def delete_file(self, storage_url: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            storage_url: Storage URL from UploadResult
            
        Returns:
            True if deleted successfully
        """
        try:
            if storage_url.startswith("s3://"):
                if self.s3_client:
                    # Extract key from s3://bucket/key
                    parts = storage_url.replace("s3://", "").split("/", 1)
                    if len(parts) == 2:
                        self.s3_client.delete_object(
                            Bucket=parts[0],
                            Key=parts[1]
                        )
                        return True
            else:
                # Local file
                if os.path.exists(storage_url):
                    os.remove(storage_url)
                    return True
        except Exception as e:
            logger.error(f"[Storage] Delete failed: {e}")
        
        return False
    
    def delete_job_files(self, job_id: str) -> bool:
        """
        Delete all files for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if deleted successfully
        """
        prefix = f"notes/{job_id}/"
        
        try:
            if self.storage_mode == "s3" and self.s3_client:
                # List and delete all objects with prefix
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    objects = page.get('Contents', [])
                    if objects:
                        self.s3_client.delete_objects(
                            Bucket=self.bucket_name,
                            Delete={
                                'Objects': [{'Key': obj['Key']} for obj in objects]
                            }
                        )
                return True
            else:
                # Local storage
                local_path = os.path.join(self.local_folder, prefix)
                if os.path.exists(local_path):
                    import shutil
                    shutil.rmtree(local_path, ignore_errors=True)
                return True
                
        except Exception as e:
            logger.error(f"[Storage] Delete job files failed: {e}")
            return False


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
