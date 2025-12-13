"""
Improved Model Training Script with Better Accuracy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from modules.ml_models_improved import ImprovedDualStageMLModel
import config

def load_data(filepath: str) -> tuple:
    """Load and prepare heart disease dataset"""
    df = pd.read_csv(filepath)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y

def train_and_save_model(data_path: str):
    """Complete training pipeline with improvements"""
    print("Loading data...")
    X, y = load_data(data_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Positive cases: {sum(y)}, Negative cases: {len(y) - sum(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, config.MODELS_DIR / 'scaler.pkl')
    print(f"Scaler saved to {config.MODELS_DIR / 'scaler.pkl'}")
    
    # Train improved model
    print("\n" + "="*50)
    print("Training IMPROVED Dual-Stage Stacked Model")
    print("="*50)
    
    model = ImprovedDualStageMLModel()
    scores = model.train(X_train_scaled, y_train)
    
    # Test evaluation
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)
    
    test_predictions = model.predict(X_test_scaled)
    test_accuracy = np.mean((test_predictions > 0.5) == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = (test_predictions > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save models
    model.save_models(config.MODELS_DIR)
    
    print("\n✅ Training completed successfully!")
    print(f"Models saved to: {config.MODELS_DIR}")
    
    return model, scores

if __name__ == "__main__":
    DATA_PATH = config.DATA_DIR / "heart_disease_data.csv"
    
    if not DATA_PATH.exists():
        print(f"❌ Dataset not found at {DATA_PATH}")
        print("Please place your heart disease dataset in the data folder")
    else:
        train_and_save_model(DATA_PATH)
