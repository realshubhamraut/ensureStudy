"""
Notes Generator Agent - Create interactive study notes
"""
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent, AgentContext
from app.rag.retriever import get_retriever


class NotesGeneratorAgent(BaseAgent):
    """
    Generates interactive study notes with key terms and definitions.
    
    Responsibilities:
    - Generate formatted study notes
    - Extract key definitions
    - Create topic summaries
    """
    
    def __init__(self):
        super().__init__(AgentContext.NOTES_GENERATOR)
        self.retriever = get_retriever()
        self.responsibilities = [
            "Generate formatted notes",
            "Extract key definitions",
            "Create study summaries"
        ]
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate interactive study notes.
        
        Args:
            input_data: {
                "topic": str,
                "subject": str,
                "difficulty": optional str
            }
        """
        self.validate_input(input_data, ["topic", "subject"])
        
        topic = input_data["topic"]
        subject = input_data["subject"]
        difficulty = input_data.get("difficulty", "medium")
        
        # Retrieve relevant content
        chunks = self.retriever.retrieve_chunks(
            query=f"{topic} {subject}",
            top_k=10,
            subject_filter=subject
        )
        
        if not chunks:
            return self.format_output({
                "title": f"{topic} - {subject}",
                "content": f"No content found for '{topic}' in {subject}.",
                "key_terms": [],
                "sources": []
            })
        
        # Format notes
        content = self._format_notes(chunks, topic)
        
        # Extract key terms
        key_terms = self._extract_definitions(chunks)
        
        # Get unique sources
        sources = list(set(c["source"] for c in chunks if c.get("source")))
        
        return self.format_output({
            "title": f"{topic} - {subject}",
            "content": content,
            "key_terms": key_terms,
            "sources": sources,
            "difficulty": difficulty,
            "chunk_count": len(chunks)
        })
    
    def _format_notes(self, chunks: List[Dict], topic: str) -> str:
        """Format retrieved chunks as structured study notes"""
        notes = f"# {topic}\n\n"
        
        # Group chunks by source
        by_source = {}
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        # Format each source section
        for source, source_chunks in by_source.items():
            # Get source name (basename)
            source_name = source.split("/")[-1] if "/" in source else source
            notes += f"## From: {source_name}\n\n"
            
            for chunk in source_chunks:
                text = chunk.get("text", "").strip()
                page = chunk.get("page", 0)
                
                if text:
                    notes += f"{text}\n\n"
                    if page:
                        notes += f"*(Page {page})*\n\n"
            
            notes += "---\n\n"
        
        return notes
    
    def _extract_definitions(self, chunks: List[Dict]) -> List[Dict[str, str]]:
        """Extract potential key terms and definitions from chunks"""
        terms = []
        definition_indicators = [
            " is ", " are ", " refers to ", " means ", " defined as ",
            " known as ", " called ", ": "
        ]
        
        seen_terms = set()
        
        for chunk in chunks:
            text = chunk.get("text", "")
            source = chunk.get("source", "")
            
            # Look for definition patterns
            for indicator in definition_indicators:
                if indicator in text.lower():
                    # Try to extract term and definition
                    sentences = text.split(". ")
                    
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            # Simple extraction - take first few words as term
                            parts = sentence.split(indicator, 1)
                            if len(parts) == 2:
                                term = parts[0].strip()[-50:]  # Last 50 chars before indicator
                                definition = parts[1].strip()[:200]  # First 200 chars after
                                
                                # Clean up term
                                term_words = term.split()[-4:]  # Last 4 words
                                term = " ".join(term_words).strip(".,;:")
                                
                                if term and term.lower() not in seen_terms and len(term) > 2:
                                    terms.append({
                                        "term": term.title(),
                                        "definition": definition,
                                        "source": source.split("/")[-1] if "/" in source else source
                                    })
                                    seen_terms.add(term.lower())
                                    
                                    if len(terms) >= 15:  # Max 15 terms
                                        return terms
        
        return terms
