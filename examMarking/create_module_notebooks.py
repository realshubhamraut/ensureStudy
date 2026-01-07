import nbformat as nbf
from pathlib import Path
import sys

# Module definitions with explanations
modules = {
    '01_Configuration': {
        'file': 'config.py',
        'title': 'Configuration Module',
        'purpose': '''This module serves as the central configuration hub for the entire evaluation system.
It defines all configurable parameters including paths, model selections, evaluation weights, and thresholds.''',
        'why': '''**Why centralize configuration?**
- Easy to modify settings without changing core logic
- Maintain consistency across modules
- Support different deployment environments
- Enable easy testing with various configurations''',
        'key_concepts': '''**Key Configuration Areas:**
1. **Directory Paths**: Project structure and file locations
2. **NLP Models**: spaCy and sentence transformer model selection
3. **Evaluation Weights**: 60% semantic, 25% keyword, 15% concept
4. **Thresholds**: Cutoff values for similarity and matching
5. **Penalties**: Rules for mark deductions'''
    },
    '02_PDF_Processor': {
        'file': 'pdf_processor.py',
        'title': 'PDF Processing Module',
        'purpose': '''Extracts text content from PDF documents submitted by students.
This is the first critical step in the evaluation pipeline.''',
        'why': '''**Why robust PDF processing matters:**
- Students submit answers as PDF files
- Need to extract plain text for NLP analysis
- Handle various PDF formats and multi-page documents
- Provide clear error messages for problematic files''',
        'key_concepts': '''**Design Decisions:**
- Uses pdfplumber (better than PyPDF2 for accuracy)
- Class-based design for state management
- Metadata extraction (pages, words, characters)
- Future-ready for OCR integration'''
    },
    '03_NLP_Preprocessor': {
        'file': 'nlp_preprocessor.py',
        'title': 'NLP Preprocessing Module',
        'purpose': '''Cleans and normalizes text through tokenization, lemmatization, and stopword removal.
Prepares text for accurate analysis by evaluation components.''',
        'why': '''**Why preprocessing is essential:**
- Raw text contains noise (punctuation, extra spaces)
- Word variations need normalization (running -> run)
- Stopwords dilute analysis importance
- Consistent format improves accuracy''',
        'key_concepts': '''**Pipeline Components:**
1. **Tokenization**: Split into sentences and words (NLTK)
2. **Lemmatization**: Convert to root forms using spaCy (more accurate than stemming)
3. **Stopword Removal**: Filter common words like "the", "is", "a"
4. **Text Cleaning**: Remove special characters and normalize whitespace
5. **Keyword Extraction**: Identify important terms by frequency'''
    },
    '04_Keyword_Matcher': {
        'file': 'keyword_matcher.py',
        'title': 'Keyword Matching Module',
        'purpose': '''Matches keywords between student and model answers with fuzzy logic support.
Handles spelling variations and typos gracefully.''',
        'why': '''**Why fuzzy matching:**
- Students make spelling mistakes
- Different but valid terminology (BST vs Binary Search Tree)
- Partial keyword matches still show understanding
- Strict exact matching is too harsh''',
        'key_concepts': '''**Matching Strategies:**
1. **Exact Match**: Perfect keyword detection
2. **Fuzzy Match**: Uses Levenshtein distance (fuzzywuzzy library)
3. **Weighted Match**: Assigns importance to different keywords
4. **Threshold**: 80% similarity by default (configurable)

**Algorithm**: For each expected keyword, find best matching student token using string similarity score.'''
    },
    '05_Semantic_Analyzer': {
        'file': 'semantic_analyzer.py',
        'title': 'Semantic Analysis Module  ',
        'purpose': '''Calculates semantic similarity using sentence transformer embeddings.
Captures conceptual understanding beyond keyword matching.''',
        'why': '''**Why semantic analysis:**
- Students might paraphrase correctly without using exact keywords
- Conceptual understanding > rote memorization
- Handles synonyms and related concepts automatically
- More aligned with how humans grade answers''',
        'key_concepts': '''**Technology:**
- Model: all-MiniLM-L6-v2 (384-dim embeddings, 80MB, fast)
- Alternative: all-mpnet-base-v2 (768-dim, 420MB, more accurate)

**Algorithm:**
1. Generate embeddings for student answer sentences
2. Generate embeddings for model answer sentences
3. Calculate cosine similarity matrix
4. Find best matches for each model sentence
5. Aggregate scores (mean of maximum similarities)

**Output**: Similarity score from 0 (completely different) to 1 (identical meaning)'''
    },
    '06_Concept_Detector': {
        'file': 'concept_detector.py',
        'title': 'Concept Detection Module',
        'purpose': '''Identifies academic concepts in student answers using knowledge base mapping.
Tracks which concepts are covered, partially covered, or missing.''',
        'why': '''**Why concept-based evaluation:**
- Ensures comprehensive answers
- Detects critical vs non-critical concept gaps
- Maps student understanding to curriculum topics
- Enables targeted feedback on what to study''',
        'key_concepts': '''**Knowledge Base Structure:**
- Concepts → Keywords mapping
- Concept weights (importance)
- Critical concept flags

**Detection Logic:**
- 70% keyword threshold for concept detection
- Partial detection (20-70% keywords found)
- Missing (<20% keywords)
- Penalty for missing critical concepts (e.g., 20% deduction)

**Example**: Concept "BST Deletion" requires keywords: deletion, remove, successor, predecessor'''
    },
    '07_Evaluation_Engine': {
        'file': 'evaluation_engine.py',
        'title': 'Evaluation Engine Module',
        'purpose': '''Orchestrates the complete evaluation by combining keyword, semantic, and concept analysis.
Calculates final scores using weighted aggregation.''',
        'why': '''**Why multi-faceted evaluation:**
- Single metric can be gamed or fail
- Hybrid approach captures different aspects
- Configurable weights allow customization
- More robust and fair than any single method''',
        'key_concepts': '''**Evaluation Formula:**
```
Final Score = (0.6 × Semantic) + (0.25 × Keyword) + (0.15 × Concept)
Marks = Final Score × Max Marks
```

**Why these weights?**
- 60% Semantic: Conceptual understanding matters most
- 25% Keyword: Technical terminology important but not everything
- 15% Concept: Ensures comprehensive coverage

**Process Flow:**
1. Preprocess student and model answers
2. Run keyword matching (parallel)
3. Run semantic analysis (parallel)
4. Run concept detection (parallel)
5. Combine scores with weights
6. Apply penalties for critical gaps
7. Convert to marks and percentage'''
    },
    '08_Feedback_Generator': {
        'file': 'feedback_generator.py',
        'title': 'Feedback Generation Module',
        'purpose': '''Generates constructive, human-readable feedback explaining scores.
Helps students understand what they did well and where to improve.''',
        'why': '''**Why detailed feedback matters:**
- Students learn from mistakes
- Transparent grading builds trust
- Actionable suggestions guide future study
- Reduces teacher workload in explaining marks''',
        'key_concepts': '''**Feedback Components:**
1. **Summary**: Marks, percentage, grade (A-F)
2. **Strengths**: What was covered well
3. **Weaknesses**: Missing concepts and keywords
4. **Mark Deductions**: Explanation for each deduction
5. **Recommendations**: Specific topics to study

**Grade Thresholds:**
- 90%+ : Excellent
- 80-89%: Very Good
- 70-79%: Good
- 60-69%: Satisfactory
- 50-59%: Pass
- <50%: Needs Improvement

**Personalization**: Feedback adapts based on specific gaps detected.'''
    },
    '09_Performance_Analyzer': {
        'file': 'performance_analyzer.py',
        'title': 'Performance Analysis Module',
        'purpose': '''Tracks student performance across multiple questions and topics.
Identifies patterns of strengths and weaknesses for personalized learning paths.''',
        'why': '''**Why performance tracking:**
- Identify topic-wise weaknesses
- Track progress over time
- Enable data-driven study plans
- Support adaptive learning systems''',
        'key_concepts': '''**Analysis Features:**
1. **Topic Aggregation**: Group questions by subject area
2. **Strength/Weakness Classification**:
   - Strong: ≥80%
   - Adequate: 60-79%
   - Weak: <60%
3. **Student Profiling**: Overall performance summary
4. **Recommendations**: Prioritized study suggestions

**Output Example:**
```json
{
  "student_id": "STU001",
  "overall_grade": "B",
  "strong_areas": ["Data Structures"],
  "weak_areas": ["Operating Systems"],
  "recommendations": [
    {"area": "OS Scheduling", "priority": "High"}
  ]
}
```

**Use Cases:**
- Teacher dashboards
- Student progress reports
- Curriculum gap analysis'''
    }
}

def create_notebook(module_id, module_info):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title and introduction
    cells.append(nbf.v4.new_markdown_cell(f"""# {module_info['title']} - Detailed Walkthrough

## Purpose

{module_info['purpose']}

## Why This Module Exists

{module_info['why']}

## Key Concepts

{module_info['key_concepts']}
"""))
    
    # Source code
    with open(f"src/{module_info['file']}", 'r', encoding='utf-8') as f:
        code = f.read()
    
    cells.append(nbf.v4.new_markdown_cell("## Complete Source Code\n\nBelow is the full implementation with inline documentation:"))
    cells.append(nbf.v4.new_code_cell(code))
    
    # Testing section
    cells.append(nbf.v4.new_markdown_cell(f"""## Testing the Module

Let's test this module to see it in action:
"""))
    
    # Add module-specific test code
    if 'config' in module_info['file']:
        test_code = """from config import Config

print("Configuration Settings:")
print(f"Semantic Weight: {Config.SEMANTIC_WEIGHT}")
print(f"Keyword Weight: {Config.KEYWORD_WEIGHT}")
print(f"Concept Weight: {Config.CONCEPT_WEIGHT}")
print(f"Transformer Model: {Config.SENTENCE_TRANSFORMER_MODEL}")"""
    
    elif 'pdf_processor' in module_info['file']:
        test_code = """from pdf_processor import PDFProcessor

processor = PDFProcessor()
# Test with sample PDF (create one first if needed)
# result = processor.extract_text('path/to/sample.pdf')
print("PDF Processor initialized successfully")
print("Ready to extract text from PDF files")"""
    
    elif 'nlp_preprocessor' in module_info['file']:
        test_code = """from nlp_preprocessor import NLPPreprocessor

preprocessor = NLPPreprocessor()
sample_text = "Binary Search Trees are efficient data structures used for searching."

processed = preprocessor.preprocess(sample_text, pipeline=['clean', 'lemmatize', 'keywords'])
print(f"Cleaned: {processed['cleaned']}")
print(f"Lemmas: {processed['lemmas']}")
print(f"Keywords: {processed['keywords']}")"""
    
    elif 'keyword_matcher' in module_info['file']:
        test_code = """from keyword_matcher import KeywordMatcher

matcher = KeywordMatcher(fuzzy_threshold=80)
student_tokens = ['bst', 'tree', 'node', 'insertion']
model_keywords = ['BST', 'binary', 'tree', 'node', 'insertion', 'deletion']

result = matcher.fuzzy_match(student_tokens, model_keywords)
print(f"Matched: {result['matched_keywords']}")
print(f"Missed: {result['missed_keywords']}")
print(f"Coverage: {result['coverage_score']:.2%}")"""
    
    elif 'semantic_analyzer' in module_info['file']:
        test_code = """from semantic_analyzer import SemanticAnalyzer

print("Loading semantic model (may take a moment)...")
analyzer = SemanticAnalyzer()

text1 = "Binary search tree is a data structure"
text2 = "BST is a hierarchical data structure"

similarity = analyzer.calculate_similarity(text1, text2)
print(f"Similarity between texts: {similarity:.3f}")"""
    
    elif 'concept_detector' in module_info['file']:
        test_code = """from concept_detector import ConceptDetector

detector = ConceptDetector()
# Load knowledge base first
# detector.load_knowledge_base('data/knowledge_base.json')

print("Concept Detector initialized")
print("Ready to detect concepts from knowledge base")"""
    
    elif 'evaluation_engine' in module_info['file']:
        test_code = """from evaluation_engine import EvaluationEngine
from config import Config

engine = EvaluationEngine(config=Config)
print("Evaluation Engine initialized")
print(f"Weights: Semantic={engine.semantic_weight}, Keyword={engine.keyword_weight}, Concept={engine.concept_weight}")"""
    
    elif 'feedback_generator' in module_info['file']:
        test_code = """from feedback_generator import FeedbackGenerator

feedback_gen = FeedbackGenerator()
grade, desc = feedback_gen.get_grade(75)
print(f"75% → Grade: {grade}, Description: {desc}")

grade, desc = feedback_gen.get_grade(92)
print(f"92% → Grade: {grade}, Description: {desc}")"""
    
    else:  # performance_analyzer
        test_code = """from performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(strong_threshold=80, weak_threshold=60)
print("Performance Analyzer initialized")
print(f"Strong threshold: 80%, Weak threshold: 60%")"""
    
    cells.append(nbf.v4.new_code_cell(test_code))
    
    # Summary
    cells.append(nbf.v4.new_markdown_cell(f"""## Summary

This module is a critical component of the AI-based answer evaluation system. It provides:

- **{module_info['title']}** functionality
- Clear, well-documented code
- Error handling and robustness
- Integration with other system modules

**Next Steps**: Explore other module notebooks to understand the complete system!
"""))
    
    nb['cells'] = cells
    return nb

# Create all notebooks
notebooks_dir = Path('notebooks')
notebooks_dir.mkdir(exist_ok=True)

print("Creating individual module notebooks...")
for module_id, module_info in modules.items():
    nb = create_notebook(module_id, module_info)
    output_file = notebooks_dir / f'{module_id}_{module_info["title"].replace(" ", "_")}.ipynb'
    with open(output_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {output_file.name}")

print(f"\\nSuccessfully created {len(modules)} notebook files in notebooks/ directory!")
