"""
Improved Dual-Stage Stacked ML Model with Hyperparameter Tuning
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from typing import Dict
import config

class ImprovedDualStageMLModel:
    """Improved dual-stage model with hyperparameter tuning"""
    
    def __init__(self):
        # Stage 1: Optimized Base Models
        self.base_models = {
            'logistic_regression': LogisticRegression(
                max_iter=2000, C=0.1, solver='liblinear', random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=10, min_samples_leaf=4, random_state=42
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            )
        }
        
        # Stage 2: Meta Model
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train all base models"""
        scores = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
            scores[name] = cv_scores.mean()
            print(f"{name} CV Score: {scores[name]:.4f}")
        
        return scores
    
    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models"""
        predictions = []
        
        for model in self.base_models.values():
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        return np.column_stack(predictions)
    
    def train_meta_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train meta model on base model predictions"""
        base_predictions = self.get_base_predictions(X_train)
        
        print("Training meta model...")
        self.meta_model.fit(base_predictions, y_train)
        self.is_trained = True
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Complete training pipeline"""
        X_train, X_meta, y_train, y_meta = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        base_scores = self.train_base_models(X_train, y_train)
        self.train_meta_model(X_meta, y_meta)
        
        final_pred = self.predict(X_meta)
        final_score = accuracy_score(y_meta, (final_pred > 0.5).astype(int))
        
        print(f"\nFinal Stacked Model Accuracy: {final_score:.4f}")
        
        return {**base_scores, 'stacked_model': final_score}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacked model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        base_predictions = self.get_base_predictions(X)
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
