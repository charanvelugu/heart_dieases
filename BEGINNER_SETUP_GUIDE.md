# 🚀 Complete Beginner Setup Guide
## For Students Who Are New to Programming

### What You Need First
1. **Python** - Download from https://python.org (Choose "Add to PATH" during installation)
2. **VS Code** - Download from https://code.visualstudio.com
3. **This project folder** - Copy the entire `heart_dieases` folder to your computer

---

## 📋 Step-by-Step Instructions

### Step 1: Open VS Code
1. Open VS Code application
2. Click `File` → `Open Folder`
3. Select the `heart_dieases` folder
4. Click `Select Folder`

### Step 2: Open Terminal in VS Code
1. In VS Code, press `Ctrl + Shift + `` (backtick key)
2. Or go to `Terminal` → `New Terminal`
3. You should see a black/white box at the bottom

### Step 3: Check Python Installation
Type this in terminal and press Enter:
```bash
python --version
```
You should see something like `Python 3.11.x` or `Python 3.12.x`

**If you get an error:**
- Download Python from https://python.org
- During installation, CHECK "Add Python to PATH"
- Restart VS Code

### Step 4: Create Virtual Environment
Copy and paste this command, then press Enter:
```bash
python -m venv venv
```
Wait for it to finish (takes 1-2 minutes)

### Step 5: Activate Virtual Environment
**Copy this EXACT command:**
```bash
venv\Scripts\activate
```
After pressing Enter, you should see `(venv)` at the start of your terminal line.

### Step 6: Install Required Packages
Copy and paste this command:
```bash
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn pytesseract Pillow PyMuPDF pdf2image google-generativeai python-dotenv joblib matplotlib seaborn python-pptx
```
This will take 3-5 minutes. Wait for it to complete.

### Step 7: Setup Environment File
1. In VS Code file explorer (left side), right-click in empty space
2. Click `New File`
3. Name it `.env` (don't forget the dot!)
4. Copy and paste this into the file:
```
GEMINI_API_KEY=your_api_key_here
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```
5. Save the file (`Ctrl + S`)

### Step 8: Get Gemini API Key (Optional for now)
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key and replace `your_api_key_here` in `.env` file

### Step 9: Train the Model
In terminal, type:
```bash
python train_model.py
```
Wait for training to complete (2-3 minutes)

### Step 10: Run the Application
In terminal, type:
```bash
streamlit run app.py
```
Your web browser should open automatically with the app!

---

## 🎯 Quick Test
1. In the web app, click "Manual Input" in sidebar
2. Enter these test values:
   - Age: 50
   - Sex: Male
   - Chest Pain: 1
   - Blood Pressure: 120
   - Cholesterol: 200
   - Fill other fields with any numbers
3. Click "Predict Risk"

---

## ❌ Common Problems & Solutions

### Problem: "python is not recognized"
**Solution:** 
- Reinstall Python from https://python.org
- CHECK "Add Python to PATH" during installation
- Restart computer

### Problem: "venv\Scripts\activate" doesn't work
**Solution:**
- Make sure you're in the right folder
- Try: `.\venv\Scripts\activate`
- Or try: `venv\Scripts\activate.bat`

### Problem: "pip install" fails
**Solution:**
- Make sure virtual environment is activated (you see `(venv)`)
- Try: `python -m pip install --upgrade pip` first
- Then run the install command again

### Problem: "Models not found"
**Solution:**
- Run `python train_model.py` first
- Wait for it to complete
- Then run `streamlit run app.py`

### Problem: Streamlit doesn't open browser
**Solution:**
- Look for a URL in terminal (like `http://localhost:8501`)
- Copy and paste it in your browser

---

## 📞 Need Help?
1. Make sure you followed each step exactly
2. Check that `(venv)` appears in your terminal
3. Ask for help with the exact error message you see

---

## 🎉 Success Signs
✅ You see `(venv)` in terminal  
✅ All packages install without errors  
✅ Training completes successfully  
✅ Web browser opens with the heart disease app  
✅ You can make predictions  

**Congratulations! You've successfully set up the project! 🎊**