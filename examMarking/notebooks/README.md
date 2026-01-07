# Module Notebooks - Individual Learning Guide

This directory contains **9 separate Jupyter notebooks**, one for each core module of the AI-based answer evaluation system.

## 📚 Notebooks Overview

Each notebook includes:
- **Purpose**: What the module does and why it exists
- **Design Rationale**: Why we made specific technical decisions
- **Key Concepts**: Important ideas and algorithms
- **Complete Source Code**: Full implementation with documentation
- **Testing Examples**: Hands-on code to see the module in action
- **Summary**: Integration with the larger system

## 📖 Learning Path

Follow these notebooks in order to understand the complete system:

### 1. **01_Configuration_Module** 
   - Central configuration hub
   - Evaluation weights and thresholds
   - Path management
   
### 2. **02_PDF_Processing_Module**
   - Text extraction from PDFs
   - Error handling
   - Metadata collection

### 3. **03_NLP_Preprocessing_Module**
   - Tokenization and lemmatization
   - Stopword removal
   - Text cleaning pipeline

### 4. **04_Keyword_Matching_Module**
   - Exact and fuzzy keyword matching
   - Spelling tolerance
   - Weighted matching

### 5. **05_Semantic_Analysis_Module**
   - Sentence transformer embeddings
   - Semantic similarity calculation
   - Conceptual understanding detection

### 6. **06_Concept_Detection_Module**
   - Knowledge base integration
   - Concept coverage analysis
   - Critical concept tracking

### 7. **07_Evaluation_Engine_Module**
   - Multi-faceted evaluation orchestration
   - Score aggregation
   - Partial marking logic

### 8. **08_Feedback_Generation_Module**
   - Constructive feedback creation
   - Mark deduction explanations
   - Study recommendations

### 9. **09_Performance_Analysis_Module**
   - Topic-wise performance tracking
   - Student profiling
   - Weakness identification

## 🚀 How to Use

1. **Install Dependencies** (if not done already):
   ```bash
   pip install -r ../requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Navigate to notebooks**:
   ```bash
   cd notebooks
   ```

3. **Open Jupyter**:
   ```bash
   jupyter notebook
   ```

4. **Work through each notebook sequentially** to build complete understanding

## 💡 Learning Tips

- **Run all cells** in each notebook to see live examples
- **Modify parameters** to see how behavior changes
- **Experiment** with the test code sections
- **Read the explanations carefully** - they explain the "why", not just the "what"

## 🎯 After Completing All Notebooks

You will understand:
- ✅ How each module works independently
- ✅ Why specific design decisions were made
- ✅ How modules integrate to form the complete system
- ✅ How to customize and extend the system

Then proceed to the main **AI_Answer_Evaluation_Complete.ipynb** in the parent directory to see the full integration!

---

**Happy Learning! 🎓**
