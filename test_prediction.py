"""
Test script for quick prediction testing
"""
import numpy as np
import joblib
from modules.ml_models import DualStageMLModel
from modules.preprocessing import DataPreprocessor
import config

def test_prediction():
    """Test prediction with sample data"""
    
    # Sample patient data
    sample_data = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    print("="*60)
    print("Heart Disease Risk Prediction - Test")
    print("="*60)
    
    # Load models
    print("\n1. Loading models...")
    try:
        model = DualStageMLModel()
        model.load_models(config.MODELS_DIR)
        scaler = joblib.load(config.MODELS_DIR / 'scaler.pkl')
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please run train_model.py first")
        return
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.prepare_features(sample_data)
    X_scaled = scaler.transform(df)
    print("✅ Data preprocessed")
    
    # Make prediction
    print("\n3. Making prediction...")
    prediction_details = model.predict_with_details(X_scaled)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\n📊 Risk Score: {prediction_details['final_prediction']:.2%}")
    print(f"🎯 Risk Level: {prediction_details['risk_level']}")
    
    print("\n🤖 Base Model Predictions:")
    for model_name, score in prediction_details['base_predictions'].items():
        print(f"   - {model_name.replace('_', ' ').title()}: {score:.2%}")
    
    print("\n" + "="*60)
    
    # Interpretation
    risk_level = prediction_details['risk_level']
    if risk_level == 'Low':
        print("✅ Low risk - Maintain healthy lifestyle")
    elif risk_level == 'Medium':
        print("⚠️ Medium risk - Consider lifestyle changes and regular checkups")
    else:
        print("🚨 High risk - Consult healthcare professional immediately")
    
    print("="*60)

if __name__ == "__main__":
    test_prediction()
