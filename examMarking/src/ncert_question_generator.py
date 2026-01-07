"""
NCERT Question Generator and Evaluator
Extracts content from NCERT PDF, generates questions, and evaluates student answers
"""

from pdf_processor import PDFProcessor
from nlp_preprocessor import NLPPreprocessor
from evaluation_engine import EvaluationEngine
from feedback_generator import FeedbackGenerator
from performance_analyzer import PerformanceAnalyzer
import json
import re
from collections import defaultdict


class NCERTQuestionGenerator:
    """Generate questions from NCERT textbook content"""
    
    def __init__(self, ncert_pdf_path):
        self.pdf_path = ncert_pdf_path
        self.pdf_processor = PDFProcessor()
        self.preprocessor = NLPPreprocessor()
        self.chapters = {}
        self.topics = {}
        
    def extract_ncert_content(self):
        """Extract text from NCERT PDF"""
        print("📚 Extracting content from NCERT textbook...")
        result = self.pdf_processor.extract_text(self.pdf_path)
        
        if result['success']:
            self.full_text = result['text']
            print(f"✓ Extracted {len(self.full_text)} characters from {result['pages']} pages")
            return True
        else:
            print(f"✗ Error: {result['error']}")
            return False
    
    def identify_chapters(self):
        """Identify chapter boundaries in NCERT text"""
        print("\n📖 Identifying chapters...")
        
        # Simple pattern: "Chapter X" or "CHAPTER X"
        chapter_pattern = r'(?:Chapter|CHAPTER)\s+(\d+)\s*[:\-]?\s*(.+?)(?=\n)'
        matches = re.finditer(chapter_pattern, self.full_text)
        
        chapters = []
        for match in matches:
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            start_pos = match.start()
            chapters.append({
                'number': int(chapter_num),
                'title': chapter_title,
                'start': start_pos
            })
        
        # Add end positions
        for i in range(len(chapters)):
            if i < len(chapters) - 1:
                chapters[i]['end'] = chapters[i+1]['start']
                chapters[i]['content'] = self.full_text[chapters[i]['start']:chapters[i]['end']]
            else:
                chapters[i]['end'] = len(self.full_text)
                chapters[i]['content'] = self.full_text[chapters[i]['start']:]
        
        self.chapters = {ch['number']: ch for ch in chapters}
        print(f"✓ Found {len(self.chapters)} chapters")
        
        for num, ch in sorted(self.chapters.items()):
            print(f"  Chapter {num}: {ch['title'][:50]}...")
        
        return self.chapters
    
    def generate_questions_from_chapter(self, chapter_num, num_questions=3):
        """Generate questions from a specific chapter"""
        if chapter_num not in self.chapters:
            print(f"✗ Chapter {chapter_num} not found")
            return []
        
        chapter = self.chapters[chapter_num]
        content = chapter['content']
        
        # Extract sentences that look like definitions or important concepts
        sentences = content.split('.')
        
        # Filter for sentences with key indicators
        important_sentences = []
        keywords = ['is defined as', 'refers to', 'means', 'is called', 
                   'is known as', 'are called', 'process of', 'method of']
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30 and len(sent) < 300:  # Reasonable length
                if any(kw in sent.lower() for kw in keywords):
                    important_sentences.append(sent)
        
        # Generate questions from important sentences
        questions = []
        for i, sent in enumerate(important_sentences[:num_questions * 2]):  # Get extra
            # Try to convert statement to question
            question_text = self._statement_to_question(sent)
            if question_text and len(questions) < num_questions:
                questions.append({
                    'id': f"CH{chapter_num}_Q{len(questions)+1}",
                    'chapter': chapter_num,
                    'chapter_title': chapter['title'],
                    'question': question_text,
                    'model_answer': sent,  # Original sentence as model answer
                    'keywords': self._extract_key_terms(sent),
                    'max_marks': 5
                })
        
        return questions
    
    def _statement_to_question(self, statement):
        """Convert a statement into a question"""
        statement = statement.strip()
        
        # Patterns to create questions
        patterns = [
            (r'(.+?)\s+is defined as\s+(.+)', 'Define {0}.'),
            (r'(.+?)\s+refers to\s+(.+)', 'What does {0} refer to?'),
            (r'(.+?)\s+means\s+(.+)', 'What does {0} mean?'),
            (r'(.+?)\s+is called\s+(.+)', 'What is {0}?'),
            (r'(.+?)\s+are called\s+(.+)', 'What are {0}?'),
            (r'The process of\s+(.+?)\s+is\s+(.+)', 'Explain the process of {0}.'),
        ]
        
        for pattern, question_template in patterns:
            match = re.search(pattern, statement, re.IGNORECASE)
            if match:
                return question_template.format(match.group(1).strip())
        
        # Default: Ask to explain
        words = statement.split()
        if len(words) > 3:
            topic = ' '.join(words[:5])
            return f"Explain {topic}..."
        
        return None
    
    def _extract_key_terms(self, text):
        """Extract key technical terms from text"""
        processed = self.preprocessor.preprocess(text, pipeline=['clean', 'lemmatize'])
        lemmas = processed['lemmas']
        
        # Filter for important words (longer, capitalized in original, etc.)
        key_terms = []
        words = text.split()
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 4 and (clean_word[0].isupper() or clean_word.isupper()):
                key_terms.append(clean_word.lower())
        
        # Add lemmas that are longer
        key_terms.extend([l for l in lemmas if len(l) > 5])
        
        return list(set(key_terms))[:15]  # Top 15 unique terms
    
    def create_knowledge_base_from_questions(self, questions):
        """Create knowledge base entry for generated questions"""
        kb = {"questions": {}}
        
        for q in questions:
            kb["questions"][q['id']] = {
                "question_text": q['question'],
                "topic": q['chapter_title'],
                "max_marks": q['max_marks'],
                "keywords": q['keywords'],
                "concepts": {
                    q['chapter_title']: {
                        "keywords": q['keywords'],
                        "weight": 1.0,
                        "critical": True
                    }
                }
            }
        
        return kb


class InteractiveExamSystem:
    """Interactive exam system with question generation and evaluation"""
    
    def __init__(self, ncert_pdf_path, knowledge_base_path=None):
        self.generator = NCERTQuestionGenerator(ncert_pdf_path)
        self.engine = EvaluationEngine()
        self.feedback_gen = FeedbackGenerator()
        self.perf_analyzer = PerformanceAnalyzer()
        self.current_questions = []
        self.student_answers = []
        
    def setup(self):
        """Setup: Extract NCERT content and identify chapters"""
        if self.generator.extract_ncert_content():
            self.generator.identify_chapters()
            return True
        return False
    
    def generate_exam(self, chapter_nums, questions_per_chapter=2):
        """Generate exam with specified parameters"""
        print(f"\n🎯 Generating exam with {questions_per_chapter} questions per chapter...")
        
        self.current_questions = []
        for ch_num in chapter_nums:
            questions = self.generator.generate_questions_from_chapter(
                ch_num, questions_per_chapter
            )
            self.current_questions.extend(questions)
        
        print(f"✓ Generated {len(self.current_questions)} questions total")
        return self.current_questions
    
    def display_questions(self):
        """Display generated questions"""
        print("\n" + "="*70)
        print("EXAM QUESTIONS")
        print("="*70)
        
        for i, q in enumerate(self.current_questions, 1):
            print(f"\nQuestion {i} [{q['max_marks']} marks]")
            print(f"Chapter: {q['chapter_title']}")
            print(f"{q['question']}")
            print("-" * 70)
    
    def accept_answer(self, question_id, answer_text=None, answer_pdf=None):
        """Accept student answer (text or PDF)"""
        if answer_pdf:
            # Extract from PDF
            processor = PDFProcessor()
            result = processor.extract_text(answer_pdf)
            if result['success']:
                answer_text = result['text']
            else:
                print(f"Error reading PDF: {result['error']}")
                return False
        
        if not answer_text:
            print("No answer provided")
            return False
        
        self.student_answers.append({
            'question_id': question_id,
            'answer': answer_text
        })
        return True
    
    def evaluate_exam(self):
        """Evaluate all answers"""
        print("\n⚙️ Evaluating answers...")
        
        # Create knowledge base
        kb = self.generator.create_knowledge_base_from_questions(self.current_questions)
        self.engine.concept_detector.knowledge_base = kb
        
        results = []
        for i, q in enumerate(self.current_questions):
            # Find corresponding answer
            answer = next((a for a in self.student_answers if a['question_id'] == q['id']), None)
            
            if answer:
                result = self.engine.evaluate(
                    student_answer=answer['answer'],
                    model_answer=q['model_answer'],
                    question_data=kb['questions'][q['id']],
                    max_marks=q['max_marks']
                )
                result['question_id'] = q['id']
                result['question_text'] = q['question']
                result['chapter'] = q['chapter_title']
                results.append(result)
        
        return results
    
    def generate_topic_feedback(self, evaluation_results):
        """Generate topic-wise performance feedback"""
        performance = self.perf_analyzer.analyze_single_exam(evaluation_results)
        
        # Group by chapter/topic
        chapter_performance = defaultdict(list)
        for result in evaluation_results:
            chapter_performance[result['chapter']].append(result['percentage'])
        
        topic_feedback = {}
        for chapter, percentages in chapter_performance.items():
            avg = sum(percentages) / len(percentages)
            topic_feedback[chapter] = {
                'average_score': avg,
                'status': 'Strong' if avg >= 80 else 'Needs Focus' if avg < 60 else 'Good',
                'questions_attempted': len(percentages)
            }
        
        return topic_feedback, performance
