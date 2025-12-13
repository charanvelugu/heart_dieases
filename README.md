# Virtual Environment Setup

## Steps to Create and Use Virtual Environment

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Verify Activation
```bash
python --version
pip --version
```

### 4. Install Dependencies (if you have requirements.txt)
```bash
pip install -r requirements.txt
```

### 5. Create requirements.txt (to save current packages)
```bash
pip freeze > requirements.txt
```

### 6. Deactivate Virtual Environment
```bash
deactivate
```

## Notes
- Always activate the virtual environment before working on the project
- The `venv` folder should be added to `.gitignore`
- Use `pip install package_name` to install new packages while venv is active