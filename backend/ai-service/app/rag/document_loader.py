"""
Document Loader and Processor for RAG Pipeline
"""
import os
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from app.rag.qdrant_setup import get_qdrant_client, ingest_documents


class DocumentProcessor:
    """Load, chunk, and embed documents for Qdrant ingestion"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: Optional[str] = None
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # Use free HuggingFace embeddings (no API key required)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.qdrant_client = get_qdrant_client()
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "classroom_materials")
    
    def load_text_file(self, file_path: str) -> str:
        """Load content from a text file"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def load_pdf_file(self, file_path: str) -> List[dict]:
        """Load content from a PDF file"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            return [
                {
                    "text": page.page_content,
                    "metadata": {
                        "source": file_path,
                        "page": page.metadata.get("page", i),
                        "type": "pdf"
                    }
                }
                for i, page in enumerate(pages)
            ]
        except ImportError:
            print("PyPDF not installed. Run: pip install pypdf")
            return []
    
    def load_documents(self, source_path: str) -> List[dict]:
        """
        Load documents from file or directory.
        
        Args:
            source_path: Path to file or directory
        
        Returns:
            List of document dicts with 'text' and 'metadata' keys
        """
        path = Path(source_path)
        documents = []
        
        if path.is_file():
            documents.extend(self._load_single_file(path))
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        documents.extend(self._load_single_file(file_path))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[dict]:
        """Load a single file based on extension"""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return self.load_pdf_file(str(file_path))
        elif suffix in [".txt", ".md", ".rst"]:
            content = self.load_text_file(str(file_path))
            return [{
                "text": content,
                "metadata": {
                    "source": str(file_path),
                    "type": "text"
                }
            }]
        else:
            return []
    
    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into chunks"""
        chunks = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            text_chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                })
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        return self.embeddings.embed_documents(texts)
    
    def process_and_ingest(
        self,
        source_path: str,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        difficulty: str = "medium"
    ) -> int:
        """
        End-to-end: load → chunk → embed → ingest to Qdrant
        
        Args:
            source_path: Path to file or directory
            subject: Subject category (e.g., "Biology", "Math")
            topic: Topic name
            difficulty: Difficulty level
        
        Returns:
            Number of chunks ingested
        """
        print(f"📚 Loading documents from {source_path}...")
        documents = self.load_documents(source_path)
        print(f"✓ Loaded {len(documents)} documents")
        
        if not documents:
            print("⚠ No documents to process")
            return 0
        
        print("✂️ Chunking documents...")
        chunks = self.chunk_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Add extra metadata
        for chunk in chunks:
            if subject:
                chunk["metadata"]["subject"] = subject
            if topic:
                chunk["metadata"]["topic"] = topic
            chunk["metadata"]["difficulty"] = difficulty
        
        print("🧠 Generating embeddings...")
        texts = [c["text"] for c in chunks]
        embeddings = self.generate_embeddings(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        print("📤 Ingesting to Qdrant...")
        count = ingest_documents(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            documents=chunks,
            embeddings=embeddings
        )
        
        return count


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Try to load sample documents
    sample_dir = "data/sample_documents"
    if Path(sample_dir).exists():
        count = processor.process_and_ingest(
            source_path=sample_dir,
            subject="General",
            topic="Sample"
        )
        print(f"✓ Ingested {count} chunks")
    else:
        print(f"Sample directory not found: {sample_dir}")
