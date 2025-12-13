# 🚀 Quick Setup Guide

Follow these steps to get the Heart Disease Prediction System running:

## ✅ Step-by-Step Setup

### Step 1: Navigate to Project Directory
```bash
cd c:\Users\reddyvel\Desktop\outside_project\heart_dieases
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r ..\requirements.txt
```

### Step 5: Install Tesseract OCR (for PDF extraction)

**Download and Install:**
- Go to: https://github.com/UB-Mannheim/tesseract/wiki
- Download the Windows installer
- Install to default location: `C:\Program Files\Tesseract-OCR`

### Step 6: Setup Gemini API Key

1. Get API key from: https://makersuite.google.com/app/apikey
2. Create `.env` file:
```bash
copy .env.example .env
```
3. Edit `.env` and add your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Step 7: Train the Models
```bash
python train_model.py
```

**Expected Output:**
- Training progress for each model
- Cross-validation scores
- Final stacked model accuracy
- Models saved to `models/` folder

### Step 8: Test the System
```bash
python test_prediction.py
```

**Expected Output:**
- Risk score and level
- Individual model predictions
- Risk interpretation

### Step 9: Run the Web Application
```bash
streamlit run app.py
```

**Expected Output:**
- Browser opens automatically at `http://localhost:8501`
- Streamlit interface loads

## 🎯 Quick Test

Once the app is running:

1. **Test Manual Input:**
   - Click "Manual Input" in sidebar
   - Enter sample values:
     - Age: 63
     - Sex: Male
     - Chest Pain: 3
     - BP: 145
     - Cholesterol: 233
     - etc.
   - Click "Predict Risk"

2. **Test PDF Upload:**
   - Click "PDF Upload"
   - Upload a medical report PDF
   - Review extracted parameters
   - Click "Predict Risk"

## 🔧 Troubleshooting

### Issue: "Models not found"
**Solution:** Run `python train_model.py` first

### Issue: "Tesseract not found"
**Solution:** 
- Install Tesseract OCR
- Update path in `.env` file

### Issue: "Gemini API error"
**Solution:**
- Check API key in `.env`
- Verify internet connection
- Check API quota at Google AI Studio

### Issue: "Module not found"
**Solution:**
- Ensure virtual environment is activated
- Run `pip install -r ..\requirements.txt` again

## 📊 Dataset Format

Your dataset should have these columns:
```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

- First 13 columns: Features
- Last column: Target (0 = No disease, 1 = Disease)

## 🎉 Success Indicators

✅ Virtual environment activated (venv in prompt)
✅ All packages installed without errors
✅ Models trained and saved in `models/` folder
✅ Test prediction runs successfully
✅ Streamlit app opens in browser
✅ Predictions work correctly

## 📞 Need Help?

Check the main README.md for detailed documentation.

## 🎯 Next Steps

After setup:
1. Test with sample data
2. Upload your own medical reports
3. Customize risk thresholds in `config.py`
4. Train with your own dataset
5. Deploy to cloud (Streamlit Cloud, AWS, etc.)

---

**Happy Predicting! **
