"""
Data Preprocessing & Feature Engineering Module
Cleans, normalizes, and prepares data for ML models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import config

class DataPreprocessor:
    """Preprocess and engineer features for heart disease prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = config.FEATURE_NAMES
    
    def fill_missing_values(self, data: Dict[str, float]) -> Dict[str, float]:
        """Fill missing values with defaults"""
        defaults = {
            'age': 50, 'sex': 1, 'cp': 0, 'trestbps': 120,
            'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 150,
            'exang': 0, 'oldpeak': 0, 'slope': 1, 'ca': 0, 'thal': 2
        }
        
        for feature in self.feature_names:
            if feature not in data or data[feature] is None:
                data[feature] = defaults.get(feature, 0)
        
        return data
    
    def validate_ranges(self, data: Dict[str, float]) -> Dict[str, float]:
        """Validate and clip values to acceptable medical ranges"""
        ranges = {
            'age': (0, 120),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (80, 200),
            'chol': (100, 600),
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (60, 220),
            'exang': (0, 1),
            'oldpeak': (0, 10),
            'slope': (0, 2),
            'ca': (0, 3),
            'thal': (0, 3)
        }
        
        for feature, (min_val, max_val) in ranges.items():
            if feature in data:
                data[feature] = np.clip(data[feature], min_val, max_val)
        
        return data
    
    def prepare_features(self, data: Dict[str, float]) -> pd.DataFrame:
        """Convert dict to DataFrame with proper feature order"""
        # Fill missing values
        data = self.fill_missing_values(data)
        
        # Validate ranges
        data = self.validate_ranges(data)
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all features are present in correct order
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[self.feature_names]
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if fit:
            return self.scaler.fit_transform(df)
        else:
            return self.scaler.transform(df)
    
    def process_input(self, raw_data: Dict[str, float]) -> np.ndarray:
        """Complete preprocessing pipeline"""
        df = self.prepare_features(raw_data)
        # Note: For prediction, scaler should be loaded from trained model
        return df.values
