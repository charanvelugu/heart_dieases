@echo off
echo ========================================
echo Heart Disease Prediction System
echo AUTOMATIC SETUP FOR BEGINNERS
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo.
echo Step 2: Creating virtual environment...
if not exist "venv\" (
    python -m venv venv
    echo ✅ Virtual environment created!
) else (
    echo ✅ Virtual environment already exists!
)

echo.
echo Step 3: Activating virtual environment...
call venv\Scripts\activate

echo.
echo Step 4: Installing required packages...
echo This may take 3-5 minutes, please wait...
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn pytesseract Pillow PyMuPDF pdf2image google-generativeai python-dotenv joblib matplotlib seaborn python-pptx

echo.
echo Step 5: Creating .env file...
if not exist ".env" (
    echo GEMINI_API_KEY=your_api_key_here > .env
    echo TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe >> .env
    echo ✅ .env file created!
) else (
    echo ✅ .env file already exists!
)

echo.
echo Step 6: Training the model...
python train_model.py

echo.
echo ========================================
echo 🎉 SETUP COMPLETE! 🎉
echo ========================================
echo.
echo To run the application:
echo 1. Make sure virtual environment is active (you should see (venv))
echo 2. Run: streamlit run app.py
echo 3. Your browser will open automatically
echo.
echo Press any key to start the application now...
pause > nul

echo.
echo Starting the application...
streamlit run app.py