"""
Streamlit Frontend for Heart Disease Prediction System
Main application interface
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from modules.pdf_ocr import PDFOCRExtractor
from modules.preprocessing import DataPreprocessor
try:
    from modules.ml_models_improved import ImprovedDualStageMLModel as DualStageMLModel
except:
    from modules.ml_models import DualStageMLModel
from modules.llm_analyzer import LLMHealthAnalyzer
import config

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Initialize session state
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

def load_models():
    """Load trained models"""
    try:
        model = DualStageMLModel()
        model.load_models(config.MODELS_DIR)
        scaler = joblib.load(config.MODELS_DIR / 'scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please train the models first by running train_model.py")
        return None, None

def main():
    # Header
    st.title("❤️ Heart Disease Risk Prediction System")
    st.markdown("### AI-Powered Medical Report Analysis")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio("Select Option", 
                       ["PDF Upload", "Manual Input", "About"])
        
        st.markdown("---")
        st.info("💡 Upload medical reports or enter parameters manually")
    
    # Load models
    model, scaler = load_models()
    
    if model is None:
        st.warning("⚠️ Models not loaded. Please train models first.")
        return
    
    # Main content
    if page == "PDF Upload":
        pdf_upload_page(model, scaler)
    elif page == "Manual Input":
        manual_input_page(model, scaler)
    else:
        about_page()

def pdf_upload_page(model, scaler):
    """PDF upload and processing page"""
    st.header("📄 Upload Medical Report (PDF)")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        # Save uploaded file
        save_path = config.UPLOAD_DIR / uploaded_file.name
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Extract data
        with st.spinner("🔍 Extracting data from PDF..."):
            ocr_extractor = PDFOCRExtractor()
            extracted_params, extracted_text = ocr_extractor.process_pdf(str(save_path))
        
        # Display extracted text
        with st.expander("📝 Extracted Text"):
            st.text_area("Raw Text", extracted_text[:1000], height=200)
        
        # Display extracted parameters
        st.subheader("🔬 Extracted Medical Parameters")
        
        if extracted_params:
            col1, col2 = st.columns(2)
            with col1:
                st.json(extracted_params)
        else:
            st.warning("⚠️ No parameters auto-extracted. Please enter manually below.")
        
        # Allow manual editing
        st.subheader("✏️ Edit/Complete Parameters")
        edited_params = manual_parameter_input(extracted_params)
        
        # Predict button
        if st.button("🔮 Predict Risk", type="primary"):
            make_prediction(edited_params, model, scaler)

def manual_input_page(model, scaler):
    """Manual parameter input page"""
    st.header("✏️ Manual Parameter Entry")
    
    params = manual_parameter_input({})
    
    if st.button("🔮 Predict Risk", type="primary"):
        make_prediction(params, model, scaler)

def manual_parameter_input(default_values: dict) -> dict:
    """Create input form for medical parameters"""
    params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        params['age'] = st.number_input("Age", 0, 120, 
                                       int(default_values.get('age', 50)))
        params['sex'] = st.selectbox("Sex", [1, 0], 
                                    format_func=lambda x: "Male" if x == 1 else "Female")
        params['cp'] = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        params['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", 
                                            80, 200, int(default_values.get('trestbps', 120)))
        params['chol'] = st.number_input("Cholesterol (mg/dl)", 
                                        100, 600, int(default_values.get('chol', 200)))
        params['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
        params['restecg'] = st.selectbox("Resting ECG Results", [0, 1, 2])
    
    with col2:
        params['thalach'] = st.number_input("Max Heart Rate", 
                                           60, 220, int(default_values.get('thalach', 150)))
        params['exang'] = st.selectbox("Exercise Induced Angina", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
        params['oldpeak'] = st.number_input("ST Depression", 
                                           0.0, 10.0, float(default_values.get('oldpeak', 0.0)))
        params['slope'] = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
        params['ca'] = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        params['thal'] = st.selectbox("Thalassemia", [0, 1, 2, 3])
    
    return params

def make_prediction(params: dict, model, scaler):
    """Make prediction and display results"""
    with st.spinner("🔄 Processing..."):
        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_features(params)
        X_scaled = scaler.transform(df)
        
        # Predict
        prediction_details = model.predict_with_details(X_scaled)
        risk_score = prediction_details['final_prediction']
        risk_level = prediction_details['risk_level']
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Risk score display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.1%}")
        with col2:
            st.metric("Risk Level", risk_level)
        with col3:
            color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
            st.metric("Status", color)
        
        # Model breakdown
        with st.expander("🤖 Model Predictions Breakdown"):
            pred_df = pd.DataFrame([prediction_details['base_predictions']])
            st.bar_chart(pred_df.T)
        
        # LLM Analysis
        st.subheader("🧠 AI Health Analysis")
        
        try:
            llm_analyzer = LLMHealthAnalyzer()
            with st.spinner("Generating personalized health report..."):
                health_report = llm_analyzer.generate_health_report(
                    params, prediction_details, risk_score
                )
            st.markdown(health_report)
        except Exception as e:
            st.warning(f"LLM analysis unavailable: {e}")
            quick_summary = f"""
            **Risk Assessment:** {risk_level} Risk ({risk_score:.1%})
            
            **Quick Summary:**
            - Your heart disease risk has been assessed as {risk_level.lower()}.
            - Please consult with a healthcare professional for detailed evaluation.
            """
            st.info(quick_summary)

def about_page():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### Heart Disease Risk Prediction System
    
    This AI-powered system helps predict heart disease risk using:
    
    **🔬 Technologies:**
    - **OCR**: Extracts data from medical PDF reports
    - **Dual-Stage ML**: Combines 5 models (Logistic Regression, Random Forest, SVM, XGBoost, Decision Tree)
    - **LLM Integration**: Gemini AI generates personalized health insights
    
    **📋 Features:**
    - PDF medical report upload and automatic extraction
    - Manual parameter entry
    - Multi-model ensemble prediction
    - AI-generated health recommendations
    
    **⚠️ Disclaimer:**
    This system is for educational and informational purposes only. 
    Always consult qualified healthcare professionals for medical advice.
    """)

if __name__ == "__main__":
    main()
