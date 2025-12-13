"""
PDF Upload & OCR Text Extraction Module
Extracts medical parameters from PDF reports using OCR
"""
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import Dict, Optional
import config

class PDFOCRExtractor:
    """Extract text and medical parameters from PDF reports"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        elif config.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error extracting text: {e}")
        return text
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR for scanned PDFs"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                text += pytesseract.image_to_string(img)
            doc.close()
        except Exception as e:
            print(f"Error in OCR: {e}")
        return text
    
    def extract_medical_parameters(self, text: str) -> Dict[str, float]:
        """Extract medical parameters from text using regex patterns"""
        params = {}
        
        # Pattern matching for common medical parameters
        patterns = {
            'age': r'age[:\s]+(\d+)',
            'trestbps': r'(?:blood pressure|bp)[:\s]+(\d+)',
            'chol': r'(?:cholesterol|chol)[:\s]+(\d+)',
            'thalach': r'(?:max heart rate|heart rate)[:\s]+(\d+)',
            'oldpeak': r'(?:st depression|oldpeak)[:\s]+(\d+\.?\d*)',
            'fbs': r'(?:fasting blood sugar|fbs)[:\s]+(\d+)',
        }
        
        text_lower = text.lower()
        
        for param, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    params[param] = float(match.group(1))
                except ValueError:
                    pass
        
        return params
    
    def process_pdf(self, pdf_path: str) -> Dict[str, float]:
        """Main method to process PDF and extract parameters"""
        # Try direct text extraction first
        text = self.extract_text_from_pdf(pdf_path)
        
        # If no text found, use OCR
        if not text.strip():
            text = self.extract_text_with_ocr(pdf_path)
        
        # Extract medical parameters
        params = self.extract_medical_parameters(text)
        
        return params, text
