"""
Celery Configuration for Background Document Processing
"""
import os
from celery import Celery

# Redis broker URL from environment
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

# Create Celery app
celery_app = Celery(
    'document_processor',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['app.workers.document_tasks']
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Rate limiting
    worker_prefetch_multiplier=1,
    
    # Task routing
    task_routes={
        'app.workers.document_tasks.process_document': {'queue': 'documents'},
        'app.workers.document_tasks.extract_pages': {'queue': 'documents'},
        'app.workers.document_tasks.ocr_page': {'queue': 'ocr'},
        'app.workers.document_tasks.chunk_and_embed': {'queue': 'documents'},
        'app.workers.document_tasks.index_to_qdrant': {'queue': 'documents'},
    },
    
    # Result expiration (1 hour)
    result_expires=3600,
    
    # Logging
    worker_hijack_root_logger=False,
)

# Task priority queues
celery_app.conf.task_queues = {
    'documents': {
        'exchange': 'documents',
        'routing_key': 'documents',
    },
    'ocr': {
        'exchange': 'ocr',
        'routing_key': 'ocr',
    },
}
