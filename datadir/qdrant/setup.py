#!/usr/bin/env python3
"""
Qdrant Collection Setup Script

Creates all required collections for ensureStudy.
Run this after starting the Qdrant container.

Usage:
    python datadir/qdrant/setup.py
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, 
    OptimizersConfigDiff, HnswConfigDiff,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType
)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))


def create_collections():
    """Create all Qdrant collections for ensureStudy."""
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    collections = [
        {
            "name": "ensure_study_documents",
            "description": "Main RAG collection for documents",
            "config": {
                "vectors_config": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
                "optimizers_config": OptimizersConfigDiff(
                    indexing_threshold=20000,
                    memmap_threshold=20000
                ),
                "hnsw_config": HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            }
        },
        {
            "name": "classroom_materials",
            "description": "Classroom-specific document chunks",
            "config": {
                "vectors_config": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
                "optimizers_config": OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=10000
                ),
                "hnsw_config": HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            }
        },
        {
            "name": "web_cache",
            "description": "Semantic web content cache",
            "config": {
                "vectors_config": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            }
        },
        {
            "name": "student_notes",
            "description": "Personal student notes",
            "config": {
                "vectors_config": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            }
        },
        {
            "name": "youtube_transcripts",
            "description": "YouTube video transcript chunks",
            "config": {
                "vectors_config": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            }
        }
    ]
    
    for coll in collections:
        name = coll["name"]
        try:
            # Check if exists
            existing = client.get_collections().collections
            existing_names = [c.name for c in existing]
            
            if name in existing_names:
                print(f"  ✓ Collection '{name}' already exists")
            else:
                client.create_collection(
                    collection_name=name,
                    **coll["config"]
                )
                print(f"  ✓ Created collection '{name}'")
        except Exception as e:
            print(f"  ✗ Error with '{name}': {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("Collection Summary:")
    print("="*50)
    for coll in client.get_collections().collections:
        info = client.get_collection(coll.name)
        print(f"  {coll.name}: {info.points_count} points")


def create_payload_indices():
    """Create payload indices for efficient filtering."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    indices = {
        "ensure_study_documents": [
            ("classroom_id", "keyword"),
            ("student_id", "keyword"),
            ("document_id", "keyword"),
            ("source_type", "keyword"),
            ("subject", "keyword"),
        ],
        "classroom_materials": [
            ("classroom_id", "keyword"),
            ("document_id", "keyword"),
            ("source_type", "keyword"),
            ("subject", "keyword"),
        ],
        "student_notes": [
            ("student_id", "keyword"),
            ("subject", "keyword"),
        ]
    }
    
    print("\nCreating payload indices...")
    for collection, fields in indices.items():
        try:
            for field_name, field_type in fields:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=field_type
                )
                print(f"  ✓ Index on {collection}.{field_name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    print("="*50)
    print("Qdrant Setup for ensureStudy")
    print("="*50)
    
    create_collections()
    create_payload_indices()
    
    print("\n✅ Qdrant setup complete!")
