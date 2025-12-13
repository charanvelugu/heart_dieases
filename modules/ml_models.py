"""
Dual-Stage Stacked Machine Learning Module
Stage 1: Base Models (Logistic Regression, Random Forest, SVM, XGBoost, Decision Tree)
Stage 2: Meta Model (Stacking/Averaging)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import List, Tuple, Dict
import config

class DualStageMLModel:
    """Dual-stage stacked ensemble model for heart disease prediction"""
    
    def __init__(self):
        # Stage 1: Base Models
        self.base_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'xgboost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        
        # Stage 2: Meta Model
        self.meta_model = LogisticRegression(random_state=42)
        self.is_trained = False
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train all base models"""
        scores = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            scores[name] = cv_scores.mean()
            print(f"{name} CV Score: {scores[name]:.4f}")
        
        return scores
    
    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models"""
        predictions = []
        
        for name, model in self.base_models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        return np.column_stack(predictions)
    
    def train_meta_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train meta model on base model predictions"""
        # Get base model predictions
        base_predictions = self.get_base_predictions(X_train)
        
        # Train meta model
        print("Training meta model...")
        self.meta_model.fit(base_predictions, y_train)
        self.is_trained = True
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Complete training pipeline"""
        # Split data for meta model training
        X_train, X_meta, y_train, y_meta = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train base models
        base_scores = self.train_base_models(X_train, y_train)
        
        # Train meta model
        self.train_meta_model(X_meta, y_meta)
        
        # Evaluate final model
        final_pred = self.predict(X_meta)
        final_score = accuracy_score(y_meta, (final_pred > 0.5).astype(int))
        
        print(f"\nFinal Stacked Model Accuracy: {final_score:.4f}")
        
        return {**base_scores, 'stacked_model': final_score}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacked model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get base predictions
        base_predictions = self.get_base_predictions(X)
        
        # Meta model prediction
        final_prediction = self.meta_model.predict_proba(base_predictions)[:, 1]
        
        return final_prediction
    
    def predict_with_details(self, X: np.ndarray) -> Dict:
        """Predict with detailed breakdown"""
        base_predictions = self.get_base_predictions(X)
        final_prediction = self.meta_model.predict_proba(base_predictions)[:, 1]
        
        details = {
            'base_predictions': {
                name: float(base_predictions[0, i]) 
                for i, name in enumerate(self.base_models.keys())
            },
            'final_prediction': float(final_prediction[0]),
            'risk_level': self._get_risk_level(final_prediction[0])
        }
        
        return details
    
    def _get_risk_level(self, score: float) -> str:
        """Convert prediction score to risk level"""
        if score < config.RISK_THRESHOLDS['low']:
            return 'Low'
        elif score < config.RISK_THRESHOLDS['medium']:
            return 'Medium'
        else:
            return 'High'
    
    def save_models(self, path: str = None):
        """Save all models"""
        if path is None:
            path = config.MODELS_DIR
        
        for name, model in self.base_models.items():
            joblib.dump(model, f"{path}/{name}.pkl")
        
        joblib.dump(self.meta_model, f"{path}/meta_model.pkl")
        print(f"Models saved to {path}")
    
    def load_models(self, path: str = None):
        """Load all models"""
        if path is None:
            path = config.MODELS_DIR
        
        for name in self.base_models.keys():
            self.base_models[name] = joblib.load(f"{path}/{name}.pkl")
        
        self.meta_model = joblib.load(f"{path}/meta_model.pkl")
        self.is_trained = True
        print(f"Models loaded from {path}")
