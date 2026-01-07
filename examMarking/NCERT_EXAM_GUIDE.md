# 🎓 NCERT Interactive Exam System - User Guide

## What is This?

An **AI-powered interactive exam system** that:
- 📚 Reads your NCERT Class 10 Science textbook
- 🎯 Generates questions automatically from any chapter
- ✍️ Accepts your answers (type or upload PDF)
- 🤖 Evaluates using AI (semantic analysis + keyword matching)
- 📊 Shows which topics you need to focus on

---

## Quick Start (3 Steps!)

### Step 1: Open the Notebook
```bash
cd c:\Users\ASUS\Desktop\examMarking
jupyter notebook NCERT_Interactive_Exam_System.ipynb
```

### Step 2: Customize Your Exam
In the notebook, modify this cell:
```python
# Which chapters? (1-16 available)
selected_chapters = [1, 2, 3]  # Change these numbers!

# How many questions per chapter?
questions_per_chapter = 2  # Change this!
```

### Step 3: Write Answers & Get Results
- Write answers directly in the notebook, OR
- Upload PDF answers
- Run evaluation to get marks and feedback!

---

## Features

### ✨ Automatic Question Generation
- Extracts important concepts from NCERT chapters
- Converts statements into questions
- Identifies key terms automatically

### 📝 Flexible Answer Input
**Option 1: Type directly**
```python
answer_1 = """
Chemical reactions are processes where...
"""
```

**Option 2: Upload PDF**
```python
exam_system.accept_answer(questions[0]['id'], answer_pdf="my_answer.pdf")
```

### 🎯 Smart Evaluation
- **60% Semantic**: Understands meaning, not just keywords
- **25% Keyword**: Checks technical terminology
- **15% Concept**: Ensures comprehensive coverage

### 📊 Topic-Wise Performance
Shows you exactly which chapters need more focus:
```
✅ Chapter 1: Chemical Reactions (85% - Strong)
⚠️ Chapter 2: Acids, Bases and Salts (65% - Good)
❌ Chapter 3: Metals and Non-metals (45% - Needs Focus)
```

---

## Available Chapters (NCERT Class 10 Science)

The system automatically detects all chapters from your PDF. Typical chapters include:
1. Chemical Reactions and Equations
2. Acids, Bases and Salts
3. Metals and Non-metals
4. Carbon and its Compounds
5. Life Processes
6. Control and Coordination
7. How do Organisms Reproduce?
8. Heredity and Evolution
9. Light - Reflection and Refraction
10. The Human Eye
11. Electricity
12. Magnetic Effects of Electric Current
... and more!

---

## Example Workflow

```python
# 1. Setup
exam_system = InteractiveExamSystem("NCERT-10-Science.pdf")
exam_system.setup()

# 2. Generate 3 questions from Chapters 1 and 2
questions = exam_system.generate_exam([1, 2], questions_per_chapter=2)

# 3. Write answers
exam_system.accept_answer(questions[0]['id'], answer_text="...")
exam_system.accept_answer(questions[1]['id'], answer_pdf="answer2.pdf")

# 4. Evaluate
results = exam_system.evaluate_exam()

# 5. See performance
topic_feedback, _ = exam_system.generate_topic_feedback(results)
```

---

## Understanding Your Results

### Marks Breakdown
```
Question 1: 4.2/5 (84%)
  - Semantic: 90% (understood concept well)
  - Keyword: 70% (missing some technical terms)
  - Concept: 80% (covered most aspects)
```

### Feedback Components
- **Strengths**: What you did well
- **Weaknesses**: What was missing
- **Mark Deductions**: Why you lost marks
- **Recommendations**: What to study next

### Performance Status
- **Strong** (≥80%): Excellent understanding
- **Good** (60-79%): Adequate, some gaps
- **Needs Focus** (<60%): Requires more study

---

## Tips for Best Results

### 📖 When Answering
1. **Be comprehensive**: Cover all aspects
2. **Use technical terms**: Include keywords
3. **Explain clearly**: Show your understanding
4. **Give examples**: Where appropriate

### 🎯 To Improve Scores
1. **Review weak chapters** shown in performance analysis
2. **Focus on missing concepts** from feedback
3. **Practice technical vocabulary**
4. **Answer all parts** of the question

---

## Customization

### Change Evaluation Weights
Edit `src/config.py`:
```python
SEMANTIC_WEIGHT = 0.60  # Adjust conceptual understanding weight
KEYWORD_WEIGHT = 0.25   # Adjust terminology weight
CONCEPT_WEIGHT = 0.15   # Adjust coverage weight
```

### Adjust Performance Thresholds
```python
analyzer = PerformanceAnalyzer(
    strong_threshold=85,  # 85%+ is strong
    weak_threshold=65     # <65% needs focus
)
```

---

## Output Files

After running, you'll get:
- **`output/ncert_exam_results.json`**: Complete results with all scores
- Detailed feedback for each question in the notebook
- Topic-wise performance breakdown
- Study recommendations

---

## Troubleshooting

### "Chapter not found"
- Make sure chapter numbers match available chapters
- Run the "View Available Chapters" cell first

### "PDF extraction failed"
- Check PDF file path is correct
- Ensure PDF is not password-protected
- Try with a smaller PDF first

### Low scores despite good answers
- Check if you're using technical terminology
- Ensure comprehensive coverage of concepts
- Review the model answer format

---

## Advanced: Adding More Textbooks

To use different NCERT books:
```python
# Just change the PDF path!
exam_system = InteractiveExamSystem("NCERT-11-Physics.pdf")
```

The system automatically adapts to any NCERT textbook!

---

## System Requirements

- Python 3.8+
- Jupyter Notebook
- 2GB RAM minimum (for transformer models)
- NCERT PDF file

---

## Support

- **Documentation**: See README.md for system overview
- **Module Details**: Check `notebooks/` for individual module explanations
- **Code**: All source in `src/` directory

---

**Created with AI-powered NLP and Machine Learning** 🤖

Start learning smarter, not harder! 🚀
