"""
Configuration file for Heart Disease Prediction System
"""
import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).parent

# Directories
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model Configuration
MODEL_NAMES = ['logistic_regression', 'random_forest', 'decision_tree', 'svm', 'xgboost']
META_MODEL_NAME = 'meta_model'

# Feature Names (Standard Medical Parameters)
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
    'ca', 'thal'
]

# Risk Thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 1.0
}

# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = 'gemini-2.5-flash-lite'

# OCR Configuration
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path
OCR_LANGUAGE = 'eng'

# Medical Parameter Mappings
MEDICAL_PARAMS = {
    'age': 'Age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)',
    'restecg': 'Resting ECG Results (0-2)',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (1=Yes, 0=No)',
    'oldpeak': 'ST Depression',
    'slope': 'Slope of Peak Exercise ST Segment (0-2)',
    'ca': 'Number of Major Vessels (0-3)',
    'thal': 'Thalassemia (0-3)'
}
