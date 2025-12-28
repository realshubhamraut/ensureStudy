"""
Question Paper Parser with Transformer Support

Uses FLAN-T5 for intelligent extraction when available,
falls back to regex patterns otherwise.
"""
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Question:
    """Single question from a paper."""
    number: str
    text: str
    marks: float
    section: Optional[str] = None
    question_type: str = "short_answer"
    expected_answer: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class QuestionPaper:
    """Complete question paper structure."""
    title: str
    subject: str
    total_marks: float
    time_limit_minutes: int
    sections: List[str] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "subject": self.subject,
            "total_marks": self.total_marks,
            "time_limit_minutes": self.time_limit_minutes,
            "sections": self.sections,
            "questions": [asdict(q) for q in self.questions]
        }


# ============================================================================
# Transformer-Based Extractor
# ============================================================================

class TransformerExtractor:
    """
    Use FLAN-T5 for intelligent question paper extraction.
    
    Features:
    - Extract subject, marks, time from paper text
    - Classify question types
    - Generate grading keywords
    - Generate model answers
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def _generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate response for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def extract_metadata(self, text: str) -> dict:
        """Extract paper metadata using the model."""
        text_snippet = text[:500]
        
        # Subject
        subject = self._generate(
            f"What subject is this exam for? Text: {text_snippet}. Answer with just the subject:"
        ).strip().capitalize() or "General"
        
        # Total marks
        marks_str = self._generate(
            f"What is the total/maximum marks? Text: {text_snippet}. Number only:"
        )
        try:
            total_marks = float(re.search(r'\d+', marks_str).group())
        except:
            total_marks = 0
        
        # Time limit
        time_str = self._generate(
            f"How many minutes for this exam? Text: {text_snippet}. Number only:"
        )
        try:
            time_limit = int(re.search(r'\d+', time_str).group())
        except:
            time_limit = 0
        
        return {"subject": subject, "total_marks": total_marks, "time_limit_minutes": time_limit}
    
    def classify_question(self, question_text: str) -> str:
        """Classify question type."""
        response = self._generate(
            f"Classify as mcq/short_answer/long_answer/numerical: {question_text}"
        ).strip().lower()
        
        valid = ['mcq', 'short_answer', 'long_answer', 'numerical', 'fill_blank']
        return response if response in valid else 'short_answer'
    
    def generate_keywords(self, question_text: str) -> List[str]:
        """Generate grading keywords for a question."""
        response = self._generate(
            f"List 3-5 keywords for correct answer: {question_text}. Comma-separated:"
        )
        return [kw.strip() for kw in response.split(',') if kw.strip()][:5]
    
    def generate_answer(self, question_text: str) -> str:
        """Generate a model answer."""
        return self._generate(
            f"Answer concisely: {question_text}"
        )


# ============================================================================
# Regex-Based Extractor (Fallback)
# ============================================================================

class RegexExtractor:
    """Regex-based extraction fallback."""
    
    MARK_PATTERNS = [
        r'\((\d+(?:\.\d+)?)\s*(?:marks?|m)\)',
        r'\[(\d+)\]',
        r'(\d+)\s*marks?$',
    ]
    
    SUBJECTS = ['physics', 'chemistry', 'biology', 'mathematics', 'math', 'english', 'science']
    
    @classmethod
    def extract_metadata(cls, text: str) -> dict:
        text_lower = text.lower()
        
        # Subject
        subject = 'General'
        for subj in cls.SUBJECTS:
            if subj in text_lower:
                subject = subj.capitalize()
                break
        
        # Marks
        marks_match = re.search(r'(?:total|max)\s*marks?\s*[:\-]?\s*(\d+)', text_lower)
        total_marks = float(marks_match.group(1)) if marks_match else 0
        
        # Time
        time_match = re.search(r'time\s*[:\-]?\s*(\d+)\s*(?:hours?|hrs?)', text_lower)
        time_limit = int(time_match.group(1)) * 60 if time_match else 0
        
        return {"subject": subject, "total_marks": total_marks, "time_limit_minutes": time_limit}
    
    @classmethod
    def extract_questions(cls, text: str) -> List[Question]:
        questions = []
        section = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Section
            sec_match = re.search(r'section\s+([a-z])', line, re.I)
            if sec_match:
                section = f"Section {sec_match.group(1).upper()}"
                continue
            
            # Question
            q_match = re.match(r'^\s*(\d+)[.)]\s+(.+)', line)
            if q_match:
                q_text = q_match.group(2)
                marks = 0
                
                for pattern in cls.MARK_PATTERNS:
                    m = re.search(pattern, q_text, re.I)
                    if m:
                        marks = float(m.group(1))
                        q_text = re.sub(pattern, '', q_text, flags=re.I).strip()
                        break
                
                # Classify
                q_type = 'short_answer'
                if any(kw in q_text.lower() for kw in ['calculate', 'find', 'solve']):
                    q_type = 'numerical'
                elif any(kw in q_text.lower() for kw in ['explain', 'describe', 'discuss']):
                    q_type = 'long_answer'
                
                questions.append(Question(
                    number=q_match.group(1),
                    text=q_text,
                    marks=marks,
                    section=section,
                    question_type=q_type
                ))
        
        return questions


# ============================================================================
# Main Parser
# ============================================================================

class QuestionPaperParser:
    """
    Complete parser using transformers or regex fallback.
    """
    
    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.transformer = None
        
        if self.use_transformers:
            try:
                self.transformer = TransformerExtractor()
            except Exception:
                self.use_transformers = False
    
    def parse(self, text: str, generate_answers: bool = False) -> QuestionPaper:
        """
        Parse question paper text.
        
        Args:
            text: Raw paper text
            generate_answers: Generate model answers (transformer only)
        """
        # Metadata
        if self.use_transformers and self.transformer:
            metadata = self.transformer.extract_metadata(text)
        else:
            metadata = RegexExtractor.extract_metadata(text)
        
        # Title
        title = "Question Paper"
        for line in text.split('\n')[:5]:
            if len(line.strip()) > 10:
                title = line.strip()[:100]
                break
        
        # Questions
        questions = RegexExtractor.extract_questions(text)
        
        # Enhance with transformer
        if self.use_transformers and self.transformer:
            for q in questions:
                q.question_type = self.transformer.classify_question(q.text)
                q.keywords = self.transformer.generate_keywords(q.text)
                if generate_answers:
                    q.expected_answer = self.transformer.generate_answer(q.text)
        
        # Total marks
        total_marks = metadata['total_marks']
        if total_marks == 0:
            total_marks = sum(q.marks for q in questions)
        
        return QuestionPaper(
            title=title,
            subject=metadata['subject'],
            total_marks=total_marks,
            time_limit_minutes=metadata['time_limit_minutes'],
            sections=list(set(q.section for q in questions if q.section)),
            questions=questions
        )
    
    def to_json(self, paper: QuestionPaper) -> str:
        return json.dumps(paper.to_dict(), indent=2)


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    test = """
    PHYSICS EXAM - Class XII
    Total Marks: 50
    Time: 2 Hours
    
    Section A
    1. State Newton's first law. (2 marks)
    2. Define acceleration. (3 marks)
    
    Section B
    3. Calculate force for 5kg at 10m/s². (5 marks)
    """
    
    parser = QuestionPaperParser(use_transformers=TRANSFORMERS_AVAILABLE)
    paper = parser.parse(test)
    
    print(f"Title: {paper.title}")
    print(f"Subject: {paper.subject}")
    print(f"Total Marks: {paper.total_marks}")
    print(f"\nQuestions:")
    for q in paper.questions:
        print(f"  Q{q.number}: {q.text[:40]}... [{q.marks}m]")
