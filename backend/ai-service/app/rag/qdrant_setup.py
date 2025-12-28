"""
Qdrant Vector Database Setup and Initialization
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    OptimizersConfigDiff,
    HnswConfigDiff
)
from typing import List, Optional


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance"""
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )


def initialize_qdrant() -> QdrantClient:
    """Initialize Qdrant vector database with collection"""
    client = get_qdrant_client()
    
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "ensure_study_documents")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
    except Exception as e:
        print(f"Error getting collections: {e}")
        collection_names = []
    
    if collection_name not in collection_names:
        # Create collection with optimized settings
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dimensions,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=20000
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        )
        print(f"✓ Created Qdrant collection: {collection_name}")
    else:
        print(f"✓ Using existing Qdrant collection: {collection_name}")
    
    return client


def ingest_documents(
    client: QdrantClient,
    collection_name: str,
    documents: List[dict],
    embeddings: List[List[float]],
    start_id: int = 0
) -> int:
    """
    Ingest documents with embeddings into Qdrant.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection
        documents: List of document dicts with 'text' and 'metadata' keys
        embeddings: List of embedding vectors
        start_id: Starting ID for points
    
    Returns:
        Number of documents ingested
    """
    points = []
    
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        metadata = doc.get("metadata", {})
        
        point = PointStruct(
            id=start_id + i,
            vector=embedding,
            payload={
                "text": doc.get("text", "")[:1000],  # Store first 1000 chars
                "full_text": doc.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
                "subject": metadata.get("subject", "general"),
                "topic": metadata.get("topic", ""),
                "difficulty": metadata.get("difficulty", "medium"),
                "type": metadata.get("type", "textbook"),
                "chunk_index": i
            }
        )
        points.append(point)
    
    # Batch upsert (100 points at a time)
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    print(f"✓ Ingested {len(points)} documents into Qdrant collection: {collection_name}")
    return len(points)


def delete_collection(collection_name: Optional[str] = None) -> bool:
    """Delete a Qdrant collection"""
    client = get_qdrant_client()
    collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "ensure_study_documents")
    
    try:
        client.delete_collection(collection_name=collection)
        print(f"✓ Deleted collection: {collection}")
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False


def get_collection_info() -> dict:
    """Get info about the main collection"""
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "ensure_study_documents")
    
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.name
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test initialization
    client = initialize_qdrant()
    print(get_collection_info())
