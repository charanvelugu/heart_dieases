"""
LLM-Based Medical Summary Module
Uses Gemini to generate personalized health reports and explanations
"""
import google.generativeai as genai
from typing import Dict
import config
import os

class LLMHealthAnalyzer:
    """Generate medical insights using Gemini LLM"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in .env file")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    def generate_health_report(self, 
                              patient_data: Dict[str, float],
                              prediction_details: Dict,
                              risk_score: float) -> str:
        """Generate comprehensive health report"""
        
        prompt = self._create_prompt(patient_data, prediction_details, risk_score)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def _create_prompt(self, 
                      patient_data: Dict[str, float],
                      prediction_details: Dict,
                      risk_score: float) -> str:
        """Create detailed prompt for Gemini"""
        
        risk_level = prediction_details.get('risk_level', 'Unknown')
        
        prompt = f"""
You are a medical AI assistant. Analyze the following heart disease risk assessment and provide a clear, personalized health report.

**Patient Medical Parameters:**
{self._format_patient_data(patient_data)}

**AI Prediction Results:**
- Risk Score: {risk_score:.2%}
- Risk Level: {risk_level}
- Model Confidence Breakdown:
{self._format_model_predictions(prediction_details.get('base_predictions', {}))}

**Instructions:**
1. Provide a clear summary of the patient's heart disease risk
2. Identify and explain the key risk factors from the medical parameters
3. Highlight any concerning values (e.g., high cholesterol, abnormal ECG)
4. Suggest lifestyle modifications and preventive measures
5. Recommend when to consult a healthcare professional
6. Keep the language simple and understandable for non-medical users

**Format the report with these sections:**
- Risk Assessment Summary
- Key Risk Factors Identified
- Medical Parameters Analysis
- Recommendations & Preventive Measures
- When to Seek Medical Attention

Generate a comprehensive yet easy-to-understand health report.
"""
        return prompt
    
    def _format_patient_data(self, data: Dict[str, float]) -> str:
        """Format patient data for prompt"""
        formatted = []
        for key, value in data.items():
            param_name = config.MEDICAL_PARAMS.get(key, key)
            formatted.append(f"- {param_name}: {value}")
        return "\n".join(formatted)
    
    def _format_model_predictions(self, predictions: Dict[str, float]) -> str:
        """Format model predictions for prompt"""
        formatted = []
        for model, score in predictions.items():
            formatted.append(f"  - {model.replace('_', ' ').title()}: {score:.2%}")
        return "\n".join(formatted)
    
    def generate_quick_summary(self, risk_level: str, risk_score: float) -> str:
        """Generate a quick summary without full LLM call"""
        summaries = {
            'Low': f"✅ Low Risk ({risk_score:.1%}): Your heart disease risk is low. Maintain healthy lifestyle habits.",
            'Medium': f"⚠️ Medium Risk ({risk_score:.1%}): Moderate risk detected. Consider lifestyle changes and regular checkups.",
            'High': f"🚨 High Risk ({risk_score:.1%}): High risk detected. Consult a healthcare professional immediately."
        }
        return summaries.get(risk_level, "Risk assessment completed.")
