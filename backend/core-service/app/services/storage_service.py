"""
Storage Service - Abstract storage layer supporting local filesystem and AWS S3
Automatically switches based on STORAGE_PROVIDER environment variable
"""
import os
import uuid
from typing import Optional, BinaryIO
from datetime import datetime, timedelta

# Storage provider configuration
STORAGE_PROVIDER = os.getenv('STORAGE_PROVIDER', 'local')  # 'local' or 's3'

# Local storage configuration
LOCAL_STORAGE_PATH = os.getenv('LOCAL_STORAGE_PATH', 
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads'))

# S3 configuration
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'ensurestudy-files')
AWS_S3_REGION = os.getenv('AWS_REGION', 'ap-south-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Lazy-load S3 client
_s3_client = None


def get_s3_client():
    """Get or create S3 client (lazy loading)"""
    global _s3_client
    if _s3_client is None:
        try:
            import boto3
            _s3_client = boto3.client(
                's3',
                region_name=AWS_S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        except ImportError:
            print("[Storage] boto3 not installed, S3 storage unavailable")
            return None
    return _s3_client


class StorageService:
    """
    Abstract storage service supporting both local filesystem and S3
    
    Usage:
        storage = StorageService()
        
        # Upload a file
        url = storage.upload_file(file_data, 'recordings', 'video.webm')
        
        # Get a URL for streaming/download
        url = storage.get_url('recordings/video.webm')
        
        # Delete a file
        storage.delete_file('recordings/video.webm')
    """
    
    def __init__(self, provider: str = None):
        self.provider = provider or STORAGE_PROVIDER
        
        if self.provider == 'local':
            os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)
            print(f"[Storage] Using local storage: {LOCAL_STORAGE_PATH}")
        elif self.provider == 's3':
            print(f"[Storage] Using S3 bucket: {AWS_S3_BUCKET}")
    
    def upload_file(
        self, 
        file_data: BinaryIO, 
        folder: str, 
        filename: str,
        content_type: str = None
    ) -> str:
        """
        Upload a file and return its storage path
        
        Args:
            file_data: File-like object or bytes
            folder: Subfolder (e.g., 'recordings', 'materials')
            filename: Name of the file
            content_type: MIME type (optional)
        
        Returns:
            Storage path/key for the file
        """
        key = f"{folder}/{filename}"
        
        if self.provider == 's3':
            return self._upload_to_s3(file_data, key, content_type)
        else:
            return self._upload_to_local(file_data, key)
    
    def _upload_to_local(self, file_data: BinaryIO, key: str) -> str:
        """Upload to local filesystem"""
        file_path = os.path.join(LOCAL_STORAGE_PATH, key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if hasattr(file_data, 'read'):
            content = file_data.read()
        else:
            content = file_data
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return key
    
    def _upload_to_s3(self, file_data: BinaryIO, key: str, content_type: str = None) -> str:
        """Upload to S3"""
        s3 = get_s3_client()
        if not s3:
            raise RuntimeError("S3 client not available")
        
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        if hasattr(file_data, 'read'):
            s3.upload_fileobj(file_data, AWS_S3_BUCKET, key, ExtraArgs=extra_args or None)
        else:
            s3.put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=file_data, **extra_args)
        
        return key
    
    def upload_from_path(self, local_path: str, folder: str, filename: str = None) -> str:
        """
        Upload a file from local path
        
        Args:
            local_path: Path to local file
            folder: Target folder in storage
            filename: Optional filename (defaults to original filename)
        
        Returns:
            Storage path/key
        """
        if not filename:
            filename = os.path.basename(local_path)
        
        key = f"{folder}/{filename}"
        
        if self.provider == 's3':
            s3 = get_s3_client()
            if not s3:
                raise RuntimeError("S3 client not available")
            s3.upload_file(local_path, AWS_S3_BUCKET, key)
        else:
            # Copy file to storage location
            import shutil
            dest_path = os.path.join(LOCAL_STORAGE_PATH, key)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(local_path, dest_path)
        
        return key
    
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Get a URL for accessing the file
        
        Args:
            key: Storage path/key
            expires_in: URL expiration in seconds (S3 only)
        
        Returns:
            URL for accessing the file
        """
        if self.provider == 's3':
            return self._get_s3_url(key, expires_in)
        else:
            return self._get_local_url(key)
    
    def _get_local_url(self, key: str) -> str:
        """Get local file URL (relative to API)"""
        # Return a path that the API can serve
        # The actual serving is handled by the recordings/files routes
        return f"/api/files/{key}"
    
    def _get_s3_url(self, key: str, expires_in: int = 3600) -> str:
        """Get pre-signed S3 URL"""
        s3 = get_s3_client()
        if not s3:
            raise RuntimeError("S3 client not available")
        
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_S3_BUCKET, 'Key': key},
            ExpiresIn=expires_in
        )
        return url
    
    def get_local_path(self, key: str) -> Optional[str]:
        """
        Get local filesystem path (for local storage or cached S3 files)
        
        For S3, downloads the file to a temp location first
        
        Returns:
            Local file path or None if not available
        """
        if self.provider == 'local':
            path = os.path.join(LOCAL_STORAGE_PATH, key)
            return path if os.path.exists(path) else None
        else:
            # For S3, download to temp file
            return self._download_from_s3(key)
    
    def _download_from_s3(self, key: str) -> Optional[str]:
        """Download file from S3 to temp location"""
        import tempfile
        
        s3 = get_s3_client()
        if not s3:
            return None
        
        try:
            # Create temp file with same extension
            ext = os.path.splitext(key)[1]
            temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            s3.download_file(AWS_S3_BUCKET, key, temp_file.name)
            return temp_file.name
        except Exception as e:
            print(f"[Storage] Failed to download from S3: {e}")
            return None
    
    def delete_file(self, key: str) -> bool:
        """
        Delete a file from storage
        
        Returns:
            True if deleted, False if not found or error
        """
        if self.provider == 's3':
            return self._delete_from_s3(key)
        else:
            return self._delete_from_local(key)
    
    def _delete_from_local(self, key: str) -> bool:
        """Delete from local filesystem"""
        file_path = os.path.join(LOCAL_STORAGE_PATH, key)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"[Storage] Failed to delete local file: {e}")
            return False
    
    def _delete_from_s3(self, key: str) -> bool:
        """Delete from S3"""
        s3 = get_s3_client()
        if not s3:
            return False
        
        try:
            s3.delete_object(Bucket=AWS_S3_BUCKET, Key=key)
            return True
        except Exception as e:
            print(f"[Storage] Failed to delete from S3: {e}")
            return False
    
    def file_exists(self, key: str) -> bool:
        """Check if a file exists in storage"""
        if self.provider == 's3':
            return self._s3_file_exists(key)
        else:
            return os.path.exists(os.path.join(LOCAL_STORAGE_PATH, key))
    
    def _s3_file_exists(self, key: str) -> bool:
        """Check if file exists in S3"""
        s3 = get_s3_client()
        if not s3:
            return False
        
        try:
            s3.head_object(Bucket=AWS_S3_BUCKET, Key=key)
            return True
        except:
            return False


# Singleton instance
storage_service = StorageService()
