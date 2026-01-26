"""
RAG & Vector Search Tools - LangGraph Agent Tools

Tools for semantic search, RAG retrieval, and vector operations.
Wraps existing services: qdrant_service.py, retrieval.py, material_indexer.py
"""
from typing import List, Dict, Any, Optional
import logging

from .base_tool import AgentTool, ToolParameter, get_tool_registry

logger = logging.getLogger(__name__)


# ============================================================================
# Vector Search Tool
# ============================================================================

async def _vector_search(
    query: str,
    top_k: int = 5,
    classroom_id: Optional[str] = None,
    min_similarity: float = 0.5
) -> Dict[str, Any]:
    """Search Qdrant vector database for similar content"""
    try:
        from app.services.qdrant_service import QdrantService
        
        service = QdrantService()
        results = await service.search(
            query=query,
            top_k=top_k,
            classroom_id=classroom_id,
            min_similarity=min_similarity
        )
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"[VECTOR-SEARCH] Error: {e}")
        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": str(e)
        }


vector_search_tool = AgentTool(
    name="vector_search",
    description="Search the Qdrant vector database for semantically similar content. Returns chunks with similarity scores.",
    func=_vector_search,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query text",
            required=True
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="Number of results to return",
            required=False,
            default=5
        ),
        ToolParameter(
            name="classroom_id",
            type="string",
            description="Optional classroom ID to filter results",
            required=False,
            default=None
        ),
        ToolParameter(
            name="min_similarity",
            type="number",
            description="Minimum similarity score (0-1)",
            required=False,
            default=0.5
        )
    ],
    category="rag"
)


# ============================================================================
# RAG Retrieve Tool
# ============================================================================

async def _rag_retrieve(
    query: str,
    top_k: int = 5,
    classroom_id: Optional[str] = None,
    include_web: bool = True
) -> Dict[str, Any]:
    """Full RAG retrieval with reranking"""
    try:
        from app.rag.retriever import get_retriever
        
        retriever = get_retriever()
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            classroom_id=classroom_id
        )
        
        # Format results
        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id", ""),
                "content": r.get("content", ""),
                "source": r.get("source_type", "document"),
                "relevance": r.get("relevance_score", 0.0),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "classroom_id": r.get("classroom_id")
            })
        
        return {
            "query": query,
            "chunks": formatted,
            "count": len(formatted),
            "context": "\n\n".join([c["content"] for c in formatted[:3]])
        }
    except Exception as e:
        logger.error(f"[RAG-RETRIEVE] Error: {e}")
        return {
            "query": query,
            "chunks": [],
            "count": 0,
            "context": "",
            "error": str(e)
        }


rag_retrieve_tool = AgentTool(
    name="rag_retrieve",
    description="Retrieve relevant document chunks for a query using RAG (Retrieval Augmented Generation). Returns chunks with context for LLM.",
    func=_rag_retrieve,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Query to retrieve relevant content for",
            required=True
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="Number of chunks to retrieve",
            required=False,
            default=5
        ),
        ToolParameter(
            name="classroom_id",
            type="string",
            description="Optional classroom ID to filter",
            required=False,
            default=None
        ),
        ToolParameter(
            name="include_web",
            type="boolean",
            description="Include web-sourced content",
            required=False,
            default=True
        )
    ],
    category="rag"
)


# ============================================================================
# Index Content Tool
# ============================================================================

async def _index_content(
    content: str,
    document_id: str,
    title: str = "",
    source_type: str = "document",
    classroom_id: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Index content into the vector database"""
    try:
        from app.services.chunking_service import create_semantic_chunks
        from app.services.qdrant_service import QdrantService
        
        # Create chunks
        chunks = create_semantic_chunks(content, max_chunk_size=500, overlap=50)
        
        # Index each chunk
        service = QdrantService()
        indexed_count = 0
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "title": title,
                "source_type": source_type,
                "classroom_id": classroom_id,
                **(metadata or {})
            }
            
            await service.upsert(
                id=chunk_id,
                text=chunk,
                metadata=chunk_metadata
            )
            indexed_count += 1
        
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_count
        }
    except Exception as e:
        logger.error(f"[INDEX-CONTENT] Error: {e}")
        return {
            "success": False,
            "document_id": document_id,
            "error": str(e)
        }


index_content_tool = AgentTool(
    name="index_content",
    description="Index text content into the vector database for later retrieval. Chunks and embeds the content.",
    func=_index_content,
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="Text content to index",
            required=True
        ),
        ToolParameter(
            name="document_id",
            type="string",
            description="Unique document identifier",
            required=True
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Document title",
            required=False,
            default=""
        ),
        ToolParameter(
            name="source_type",
            type="string",
            description="Type of source (document, web, pdf, etc.)",
            required=False,
            default="document"
        ),
        ToolParameter(
            name="classroom_id",
            type="string",
            description="Classroom ID for filtering",
            required=False,
            default=None
        )
    ],
    category="rag"
)


# ============================================================================
# Delete Content Tool
# ============================================================================

async def _delete_content(document_id: str) -> Dict[str, Any]:
    """Delete content from vector database by document ID"""
    try:
        from app.services.qdrant_service import QdrantService
        
        service = QdrantService()
        deleted = await service.delete_by_document_id(document_id)
        
        return {
            "success": True,
            "document_id": document_id,
            "deleted_count": deleted
        }
    except Exception as e:
        return {
            "success": False,
            "document_id": document_id,
            "error": str(e)
        }


delete_content_tool = AgentTool(
    name="delete_content",
    description="Delete indexed content from the vector database by document ID.",
    func=_delete_content,
    parameters=[
        ToolParameter(
            name="document_id",
            type="string",
            description="Document ID to delete",
            required=True
        )
    ],
    category="rag"
)


# ============================================================================
# Register All Tools
# ============================================================================

def register_rag_tools():
    """Register all RAG tools with the global registry"""
    registry = get_tool_registry()
    
    registry.register(vector_search_tool)
    registry.register(rag_retrieve_tool)
    registry.register(index_content_tool)
    registry.register(delete_content_tool)
    
    logger.info(f"[RAG-TOOLS] Registered {len(registry.list_tools('rag'))} RAG tools")


# Auto-register on import
register_rag_tools()
