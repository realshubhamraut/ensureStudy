"""
Student Performance ML Pipeline
Uses Kaggle datasets for training models to predict student performance

Recommended Kaggle Datasets:
1. Student Performance Dataset - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
2. Open University Learning Analytics (OULAD) - https://www.kaggle.com/datasets/rocki37/open-university-learning-analytics-dataset
3. Student Study Performance - https://www.kaggle.com/datasets/bhavikjikadara/student-study-performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Create models directory
os.makedirs('models', exist_ok=True)


def load_student_performance_data(filepath='data/student_performance.csv'):
    """
    Load and preprocess student performance dataset
    Expected columns: study_hours, attendance, previous_scores, 
                     parental_education, internet_access, extracurricular, score
    """
    # If file doesn't exist, create synthetic data for demo
    if not os.path.exists(filepath):
        print("Creating synthetic dataset for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'study_hours': np.random.uniform(1, 10, n_samples),
            'attendance_rate': np.random.uniform(50, 100, n_samples),
            'previous_score': np.random.uniform(30, 100, n_samples),
            'parental_education': np.random.choice(['high_school', 'bachelors', 'masters'], n_samples),
            'internet_access': np.random.choice([0, 1], n_samples),
            'extracurricular': np.random.choice([0, 1], n_samples),
            'sleep_hours': np.random.uniform(4, 10, n_samples),
            'screen_time': np.random.uniform(1, 8, n_samples),
        }
        
        # Create target variable with realistic relationships
        target = (
            data['study_hours'] * 5 +
            data['attendance_rate'] * 0.3 +
            data['previous_score'] * 0.4 +
            data['internet_access'] * 5 +
            data['extracurricular'] * 3 +
            data['sleep_hours'] * 2 -
            data['screen_time'] * 1.5 +
            np.random.normal(0, 5, n_samples)
        )
        data['final_score'] = np.clip(target, 0, 100)
        
        df = pd.DataFrame(data)
        os.makedirs('data', exist_ok=True)
        df.to_csv(filepath, index=False)
        return df
    
    return pd.read_csv(filepath)


def eda_student_performance(df):
    """Exploratory Data Analysis on student performance data"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nCorrelation with target (final_score):")
    for col in numeric_cols:
        if col != 'final_score':
            corr = df[col].corr(df['final_score'])
            print(f"  {col}: {corr:.3f}")
    
    return df


def preprocess_data(df, target_col='final_score'):
    """Preprocess data for ML training"""
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, X.columns.tolist()


def train_regression_models(X_train, X_test, y_train, y_test):
    """Train multiple regression models and compare performance"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    best_model = None
    best_r2 = -np.inf
    
    print("\n=== MODEL TRAINING RESULTS ===\n")
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        print(f"{name}:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE:  {mae:.3f}")
        print(f"  R²:   {r2:.3f}")
        print()
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = (name, model)
    
    return results, best_model


def save_model(model, scaler, label_encoders, feature_names, model_name='student_performance_model'):
    """Save trained model and preprocessing objects"""
    model_path = f'models/{model_name}.joblib'
    scaler_path = f'models/{model_name}_scaler.joblib'
    encoders_path = f'models/{model_name}_encoders.joblib'
    features_path = f'models/{model_name}_features.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)
    joblib.dump(feature_names, features_path)
    
    print(f"\nModel saved to: {model_path}")
    return model_path


def predict_student_score(model_path, student_data):
    """Load model and make prediction for a new student"""
    model = joblib.load(model_path)
    scaler = joblib.load(model_path.replace('.joblib', '_scaler.joblib'))
    encoders = joblib.load(model_path.replace('.joblib', '_encoders.joblib'))
    features = joblib.load(model_path.replace('.joblib', '_features.joblib'))
    
    # Prepare data
    df = pd.DataFrame([student_data])
    
    # Encode categorical
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    # Scale
    X = scaler.transform(df[features])
    
    # Predict
    prediction = model.predict(X)[0]
    return float(prediction)


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("STUDENT PERFORMANCE PREDICTION PIPELINE")
    print("=" * 50)
    
    # Load data
    df = load_student_performance_data()
    
    # EDA
    df = eda_student_performance(df)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, encoders, features = preprocess_data(df)
    
    # Train models
    results, (best_name, best_model) = train_regression_models(X_train, X_test, y_train, y_test)
    
    print(f"\n=== BEST MODEL: {best_name} ===")
    
    # Save best model
    model_path = save_model(best_model, scaler, encoders, features)
    
    # Demo prediction
    print("\n=== DEMO PREDICTION ===")
    sample_student = {
        'study_hours': 7,
        'attendance_rate': 85,
        'previous_score': 75,
        'parental_education': 'bachelors',
        'internet_access': 1,
        'extracurricular': 1,
        'sleep_hours': 7,
        'screen_time': 3
    }
    
    predicted_score = predict_student_score(model_path, sample_student)
    print(f"Sample Student Data: {sample_student}")
    print(f"Predicted Score: {predicted_score:.1f}")
    
    return model_path


if __name__ == '__main__':
    main()
