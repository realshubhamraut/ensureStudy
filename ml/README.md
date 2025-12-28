# ensureStudy ML Pipeline

This directory contains machine learning and deep learning models for the ensureStudy platform.

## Kaggle Datasets for Training

We use the following educational datasets from Kaggle:

### 1. Student Performance Dataset
- **URL**: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
- **Use Case**: Predict student exam scores based on demographics and study habits
- **Features**: Gender, race/ethnicity, parental education, test preparation, scores

### 2. Open University Learning Analytics (OULAD)
- **URL**: https://www.kaggle.com/datasets/rocki37/open-university-learning-analytics-dataset
- **Use Case**: Analyze VLE interactions to predict course outcomes
- **Features**: Demographics, assessment marks, VLE clicks, module info

### 3. Student Study Performance
- **URL**: https://www.kaggle.com/datasets/bhavikjikadara/student-study-performance
- **Use Case**: Study hours and learning behavior impact on performance
- **Features**: Study hours, attendance, previous scores, activities

### 4. Multidimensional Learning Behavior
- **URL**: https://www.kaggle.com/datasets/... (search: learning behavior analytics)
- **Use Case**: Adaptive learning and personalization
- **Features**: Behavioral engagement, interaction patterns, performance outcomes

## Models

### ML Regression Models (`student_performance_pipeline.py`)
- Linear Regression
- Ridge/Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Deep Learning Models (`deep_learning_models.py`)
- Student Engagement Predictor (Neural Network)
- Content Recommendation (Neural Collaborative Filtering)
- Difficulty Level Predictor

## Usage

### Train ML Models
```bash
cd ml
pip install pandas numpy scikit-learn joblib
python student_performance_pipeline.py
```

### Train DL Models
```bash
pip install torch
python deep_learning_models.py
```

### Download Kaggle Datasets
```bash
# Install Kaggle CLI
pip install kaggle

# Set up API credentials
# Place kaggle.json in ~/.kaggle/

# Download datasets
kaggle datasets download -d spscientist/students-performance-in-exams
kaggle datasets download -d rocki37/open-university-learning-analytics-dataset
```

## Integration with AI Service

The trained models can be loaded in the AI service for:
1. **Score Prediction**: Predict expected performance on assessments
2. **Difficulty Adaptation**: Adjust question difficulty based on student level
3. **Content Recommendations**: Suggest next topics/resources
4. **Engagement Scoring**: Identify at-risk students needing intervention
