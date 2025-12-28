
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.material_indexer import MaterialIndexer, IndexResult
from app.services.qdrant_service import SourceType, ChunkMetadata
from app.api.schemas.tutor import RetrievedChunk, SourceInfo
from app.api.routes.tutor import process_tutor_query
from app.api.schemas.tutor import TutorQueryRequest, ResponseMode

@pytest.fixture
def mock_pdf_processor():
    with patch('app.services.material_indexer.PDFProcessor') as mock:
        processor = mock.return_value
        result = MagicMock()
        result.success = True
        result.total_words = 100
        result.total_pages = 1
        processor.extract_from_url.return_value = result
        processor.get_full_text.return_value = "Page 1 Content [Page 1]"
        yield processor

@pytest.fixture
def mock_qdrant_service():
    with patch('app.services.material_indexer.QdrantService') as mock:
        service = mock.return_value
        service.index_batch.return_value = ["point_1"]
        yield service

@pytest.fixture
def mock_sentence_transformer():
    with patch('sentence_transformers.SentenceTransformer') as mock:
        model = mock.return_value
        # Mock encode to return object with tolist()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        model.encode.return_value = mock_embedding
        yield model

@pytest.mark.asyncio
async def test_material_indexer_saves_url(mock_pdf_processor, mock_qdrant_service, mock_sentence_transformer):
    """Verify MaterialIndexer saves URL in metadata"""
    indexer = MaterialIndexer()
    
    # Mock extract_page_number to return 1
    indexer._extract_page_number = MagicMock(return_value=1)
    
    # Configure embedding mock to return list for batch encoding
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1] * 384
    mock_sentence_transformer.encode.return_value = [mock_embedding]
    
    result = await indexer.index_material(
        material_id="mat_123",
        file_url="http://example.com/doc.pdf",
        classroom_id="cls_456",
        subject="Biology",
        document_title="Biology 101"
    )
    
    assert result.success is True
    assert result.chunks_indexed == 1
    
    # Verify Qdrant service called with correct metadata
    mock_qdrant_service.index_batch.assert_called_once()
    batch = mock_qdrant_service.index_batch.call_args[0][0]
    embedding, metadata = batch[0]
    
    assert isinstance(metadata, ChunkMetadata)
    assert metadata.document_id == "mat_123"
    assert metadata.classroom_id == "cls_456"
    assert metadata.url == "http://example.com/doc.pdf"  # This is the key verification
    assert metadata.title == "Biology 101"

@pytest.mark.asyncio
async def test_search_classroom_materials_returns_url(mock_qdrant_service, mock_sentence_transformer):
    """Verify search_classroom_materials returns URL in dict"""
    indexer = MaterialIndexer()
    
    # Mock Qdrant search result
    mock_result = MagicMock()
    mock_result.final_score = 0.9
    mock_result.payload = {
        "chunk_text": "Sample text",
        "document_id": "doc_1",
        "title": "Doc Title",
        "page_number": 5,
        "url": "http://example.com/doc.pdf",
        "subject": "Bio"
    }
    mock_qdrant_service.search_semantic.return_value = [mock_result]
    
    results = indexer.search_classroom_materials("query", "cls_1")
    
    assert len(results) == 1
    assert results[0]["url"] == "http://example.com/doc.pdf"
    assert results[0]["page_number"] == 5

@pytest.mark.asyncio
async def test_tutor_pipeline_integration():
    """Verify tutor pipeline maps URL to response"""
    
    # Mock dependencies
    with patch('app.api.routes.tutor.semantic_search') as mock_semantic:
        with patch('app.services.material_indexer.get_material_indexer') as mock_get_indexer:
            with patch('app.api.routes.tutor.generate_answer') as mock_llm:
                with patch('app.api.routes.tutor.build_context') as mock_context:
                    
                    # Setup mocks
                    mock_semantic.return_value = []
                    
                    # Mock classroom results
                    indexer = MagicMock()
                    indexer.search_classroom_materials.return_value = [{
                        "document_id": "doc_1",
                        "chunk_text": "Relevant text",
                        "similarity_score": 0.8,
                        "title": "Classroom Doc",
                        "page_number": 2,
                        "url": "http://example.com/doc.pdf"
                    }]
                    mock_get_indexer.return_value = indexer
                    
                    # Mock LLM response
                    mock_llm.return_value = MagicMock(
                        answer_short="Answer", 
                        answer_detailed="Detail", 
                        confidence=0.9,
                        suggested_topics=[]
                    )
                    
                    # Mock context builder to return chunks used with URL
                    # Note: process_tutor_query re-constructs RetrievedChunk from classroom_results
                    # and passes it to build_context. We need to verify what happens AFTER.
                    
                    # Actually, we need to mock build_context to return an AssembledContext 
                    # that contains the chunks passed to it.
                    def side_effect_build_context(retrieved_chunks, response_mode):
                        from app.api.schemas.tutor import AssembledContext
                        return AssembledContext(
                            context_text="Context",
                            chunks_used=retrieved_chunks,
                            total_tokens=100
                        )
                    mock_context.side_effect = side_effect_build_context

                    # Execute request
                    request = TutorQueryRequest(
                        user_id="user_1",
                        question="Test question",
                        classroom_id="cls_1",
                        response_mode=ResponseMode.SHORT,
                        find_resources=False
                    )
                    
                    response = await process_tutor_query(request)
                    
                    assert response.data is not None
                    sources = response.data.sources
                    assert len(sources) > 0
                    assert sources[0].url == "http://example.com/doc.pdf"
                    assert sources[0].page_number == 2
