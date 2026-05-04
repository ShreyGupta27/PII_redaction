import os
import sys
import io
import re
import json
import logging
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import pdf2image
import img2pdf
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# PATH CONFIGURATION
if os.name == 'nt':
    tesseract_cmd = shutil.which("tesseract")
    if not tesseract_cmd:
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for path in common_paths:
            if os.path.exists(path):
                tesseract_cmd = path
                break
    
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        if 'TESSDATA_PREFIX' not in os.environ:
            tess_dir = os.path.join(os.path.dirname(tesseract_cmd), 'tessdata')
            if os.path.exists(tess_dir):
                os.environ['TESSDATA_PREFIX'] = tess_dir
    else:
        logging.warning("Tesseract-OCR not found. Please install it and add to PATH.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# CONFIGURATION
@dataclass
class Config:
    """Configuration for PII detection and redaction"""
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma2:2b-instruct-q8_0"
    timeout: int = 120
    dpi: int = 300
    redaction_color: Tuple[int, int, int] = (0, 0, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    ocr_similarity_threshold: float = 0.60  # Lowered from 0.65 for better matching

def load_biomarkers() -> set:
    """Load biomarker terms that should NOT be redacted"""
    biomarkers = set()
    biomarker_file = Path("biomarkers_example.json")
    if biomarker_file.exists():
        try:
            with open(biomarker_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "biomarkers" in data:
                    biomarkers = {term.lower().strip() for term in data["biomarkers"]}
                elif isinstance(data, list):
                    biomarkers = {term.lower().strip() for term in data}
            logger.info(f"Loaded {len(biomarkers)} biomarker terms")
        except Exception as e:
            logger.warning(f"Could not load biomarkers: {e}")
    return biomarkers

BIOMARKERS = load_biomarkers()

# Medical terms that should NEVER be redacted (exact matches only)
MEDICAL_TERMS_EXACT = {
    # Test names
    'haemoglobin', 'hemoglobin', 'sodium', 'potassium', 'chloride', 'calcium',
    'magnesium', 'phosphorus', 'urea', 'creatinine', 'bilirubin', 'albumin',
    'globulin', 'protein', 'glucose', 'cholesterol', 'triglyceride', 'hdl', 'ldl',
    'vldl', 'sgot', 'sgpt', 'ast', 'alt', 'alp', 'ggt', 'amylase', 'lipase',
    'uric', 'acid', 'iron', 'ferritin', 'tibc', 'vitamin', 'folate', 'b12',
    'tsh', 't3', 't4', 'ft3', 'ft4', 'hba1c', 'insulin', 'cortisol',
    'testosterone', 'estrogen', 'progesterone', 'prolactin', 'fsh', 'lh',
    'psa', 'cea', 'afp', 'ca-125', 'ca19-9', 'troponin', 'bnp', 'crp',
    'esr', 'wbc', 'rbc', 'platelet', 'mcv', 'mch', 'mchc', 'rdw', 'mpv',
    'neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil',
    'hematocrit', 'hct', 'pcv', 'reticulocyte', 'pt', 'inr', 'aptt', 'fibrinogen',
    'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils',
    'leucocytes', 'leukocytes', 'immature', 'granulocyte', 'percentage',
    # Section headers
    'biochemistry', 'hematology', 'serology', 'microbiology', 'urinalysis',
    'haematology', 'chemistry', 'hormone', 'specialised', 'specialized',
    'liver', 'function', 'kidney', 'lipid', 'profile', 'thyroid',
    'complete', 'blood', 'count', 'cbc', 'electrolyte', 'panel', 'test', 'report',
    'parameter', 'value', 'unit', 'reference', 'range', 'normal', 'abnormal',
    'high', 'low', 'result', 'interpretation', 'method', 'specimen', 'serum',
    'plasma', 'urine', 'sample', 'collected', 'reported', 'authorized',
    'printed', 'registered', 'pathologist', 'technician', 'laboratory',
    'ultrasensitive', 'absolute', 'total', 'final', 'status',
    # Column headers and report terms
    'biological', 'interval', 'units', 'results', 'methodology',
    'chemiluminescence', 'generation', 'method',
    # Common medical words
    'patient', 'test', 'report', 'diagnostic', 'centre', 'center', 'hospital',
    'clinic', 'lab', 'laboratory', 'medical', 'health', 'healthcare',
    'end', 'of', 'page', 'note', 'disclaimer', 'results', 'relate',
    'with', 'check', 'up', 'director', 'consultant',
    # Report-specific terms
    'presumed', 'relates', 'identified', 'requisition', 'turnaround',
    'stated', 'directory', 'quality', 'standards', 'assayed',
    'clinical', 'safety', 'technical', 'integrity', 'requested',
    'specimen', 'appropriate', 'unsatisfactory', 'insufficient',
    'ordering', 'discrepancy', 'label', 'container', 'form',
    'delays', 'circumstances', 'failure', 'parameters', 'marked',
    'asterisks', 'excluded', 'nabl', 'accredited', 'laboratory',
    'valid', 'medico', 'legal', 'purposes', 'queries', 'unexpected',
    'customer', 'care', 'call', 'investigation', 'carried',
}

# PII DETECTION PATTERNS
PII_PATTERNS = {
    "PATIENT_NAME": [
        re.compile(r'Patient\s*Name\s*[:\-]\s*(?:MR\.|MRS\.|MS\.|DR\.)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})', re.IGNORECASE),
        re.compile(r'PATIENT\s*NAME\s*[:\-]\s*(?:MR\.|MRS\.|MS\.|DR\.)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', re.IGNORECASE),
        re.compile(r'(?:^|\n)\s*Name\s*[:\-]\s*(?:MR\.|MRS\.|MS\.|DR\.)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})', re.IGNORECASE | re.MULTILINE),
        re.compile(r'\b((?:S/O|D/O|W/O)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', re.IGNORECASE),
        re.compile(r'\b((?:MR\.|MRS\.|MS\.)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b', re.IGNORECASE),
    ],
    "AGE_GENDER": [
        re.compile(r'\b(\d{1,3}\s*(?:Years?|Y|YRS?)\s*[/]\s*(?:Male|Female|M|F))\b', re.IGNORECASE),
        re.compile(r'\((\d{1,3}\s*[Y]\s*/\s*[MF])\)', re.IGNORECASE),
    ],
    "AGE": [
        re.compile(r'Age\s*[:\-]\s*(\d{1,3})\s*(?:Years?|Y|YRS?)', re.IGNORECASE),
        re.compile(r'Age[/\s]*Sex\s*[:\-]\s*(\d{1,3})\s*(?:Years?|Y|YRS?)', re.IGNORECASE),
        re.compile(r'AGE\s*[:\-]\s*(\d{1,3})\s*(?:Years?|Y|YRS?)', re.IGNORECASE),
    ],
    "GENDER": [
        re.compile(r'(?:Sex|Gender)\s*[:\-]\s*(Male|Female|M|F)\b', re.IGNORECASE),
        re.compile(r'SEX\s*[:\-]\s*(Male|Female|M|F)\b', re.IGNORECASE),
    ],
    "DOB": [
        re.compile(r'(?:DOB|Date\s*of\s*Birth|D\.O\.B\.?)\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', re.IGNORECASE),
    ],
    "LAB_NO": [
        re.compile(r'Lab\.?\s*No\.?\s*[:\-]\s*([A-Z0-9]{6,20})', re.IGNORECASE),
    ],
    "ACCESSION_NO": [
        re.compile(r'ACCESSION\s*NO\.?\s*[:\-]\s*([A-Z0-9]{8,20})', re.IGNORECASE),
        re.compile(r'\b(\d{4}[A-Z]{2}\d{6,10})\b'),
    ],
    "CLIENT_CODE": [
        re.compile(r'Client\s*(?:Code|ID)\s*[:\-]\s*([A-Z]?\d{6,15})', re.IGNORECASE),
        re.compile(r'CLIENT\s*CODE\s*[:\-]\s*([A-Z]?\d{6,15})', re.IGNORECASE),
    ],
    "PATIENT_REF": [
        re.compile(r'Patient\s*(?:Ref\.?|ID)\s*(?:No\.?)?\s*[:\-]\s*([A-Z0-9]{6,20})', re.IGNORECASE),
        re.compile(r'PATIENT\s*ID\s*[:\-]\s*([A-Z0-9]{6,20})', re.IGNORECASE),
        re.compile(r'Patient\s*Ref\.\s*No\.\s*(\d{10,15})', re.IGNORECASE),
        re.compile(r'(?:UHID|MRN)\s*[:\-]\s*([A-Z0-9]{6,20})', re.IGNORECASE),
        re.compile(r'CLIENT\s*PATIENT\s*ID\s*[:\-]\s*([A-Z0-9]{6,20})', re.IGNORECASE),
    ],
    "LIS_NUMBER": [
        re.compile(r'LIS\s*Number\s*[:\-]\s*(\d{6,15})', re.IGNORECASE),
    ],
    "CASE_NUMBER": [
        re.compile(r'Case\s*Number\s*[:\-]\s*([A-Z0-9]{6,20})', re.IGNORECASE),
    ],
    "DATE": [
        re.compile(r'(?:Registered|Collected|Authorized|Printed|Received|Reported|Drawn|Report|Sample\s*(?:Coll\.|Collected?|Rec\.))\s*(?:On|Date)?\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?)', re.IGNORECASE),
        re.compile(r'DRAWN\s*(?:ON)?\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?)', re.IGNORECASE),
        re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}\s+\d{1,2}:\d{2}\s*(?:AM|PM))\b', re.IGNORECASE),
    ],
    "PHONE": [
        re.compile(r'\b([6-9]\d{9})\b'),
        re.compile(r'(?:Tel|Fax|Phone|Mobile|Ph\.?)\s*[:\-]?\s*(\+?[\d\s\-]{10,15})', re.IGNORECASE),
        re.compile(r'Fax\s*[:\-]\s*([A-Z0-9\s\-]+)', re.IGNORECASE),
    ],
    "EMAIL": [
        re.compile(r'\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b'),
    ],
    "CLIENT_NAME": [
        re.compile(r"CLIENT'?S?\s*NAME\s*(?:AND\s*ADDRESS)?\s*[:\-]\s*([A-Z][A-Za-z\s\.&]+(?:PVT|LTD|LLC|INC|TECHNOLOGIES|HEALTHCARE|DIAGNOSTIC|CENTRE|CENTER|HOSPITAL|CLINIC|LABORATORY|LABS?)?\.?\s*(?:LTD|PVT)?\.?)", re.IGNORECASE),
        re.compile(r'Client\s*Name\s*[:\-]\s*([A-Z][A-Za-z\s\.&]+(?:PVT|LTD|TECHNOLOGIES|HEALTHCARE|DIAGNOSTIC|HOSPITAL|CLINIC)\.?\s*(?:LTD|PVT)?\.?)', re.IGNORECASE),
    ],
    "CLIENT_ADDRESS": [
        re.compile(r"CLIENT'?S?\s*NAME\s*AND\s*ADDRESS\s*[:\-]\s*([\s\S]{20,400}?)(?=\s*(?:SRL|REFERENCE\s*LAB|LAB\s*ADDRESS|PROCESSED\s*AT|Tel\s*:|PATIENT\s*NAME)|$)", re.IGNORECASE),
        re.compile(r'\b([A-Z][A-Za-z\s,]+(?:NAGAR|ROAD|STREET|COLONY|SECTOR|FLOOR|BUILDING|TOWER|APARTMENT|COMPLEX|PARK|VIHAR|ESTATE|ENCLAVE)[A-Za-z\s,]*,?\s*[A-Z][A-Za-z\s]+\s*\d{6})\b', re.IGNORECASE),
        re.compile(r'\b([A-Z][A-Za-z]+\s*,?\s*\d{6})\b'),
        re.compile(r'\b([A-Z][A-Za-z\s]+(?:BANGALORE|BENGALURU|DELHI|MUMBAI|CHENNAI|HYDERABAD|PUNE|KOLKATA|GURGAON|NOIDA|GURUGRAM)[A-Za-z\s]*\s*\d{6})\b', re.IGNORECASE),
        re.compile(r'\b([A-Z][A-Z\s]+(?:NAGAR|ROAD|COLONY|SECTOR|VIHAR|ESTATE|PARK|ENCLAVE),)\b', re.IGNORECASE),
        re.compile(r'\b([A-Z]+\s+INDIA)\b', re.IGNORECASE),
        re.compile(r'\b(KARNATAKA|HARYANA|MAHARASHTRA|TAMIL\s*NADU|DELHI|WEST\s*BENGAL|GUJARAT|RAJASTHAN|PUNJAB|UTTAR\s*PRADESH)\s*,?\s*INDIA\b', re.IGNORECASE),
    ],
    "LAB_ADDRESS": [
        re.compile(r'\b((?:SRL\s*)?REFERENCE\s*LAB[A-Za-z\s,\.]*(?:GP|MARUTI|INDUSTRIAL|ESTATE|SECTOR|VIHAR|UDYOG)[A-Za-z\s,\.\-0-9]*\d{6})\b', re.IGNORECASE),
        re.compile(r'\b([A-Z\-0-9]+,\s*MARUTI\s*INDUSTRIAL\s*ESTATE[A-Za-z\s,\.]*\d{6})\b', re.IGNORECASE),
        re.compile(r'\b(UDYOG\s*VIHAR[A-Za-z\s,\.]*SECTOR[A-Za-z\s,\.\-0-9]*\d{6})\b', re.IGNORECASE),
        re.compile(r'\b(ERENCE\s*LAB[A-Za-z\s,\.\-0-9]*(?:GP|MARUTI|INDUSTRIAL|ESTATE|UDYOG)[A-Za-z\s,\.\-0-9]*)\b', re.IGNORECASE),
        re.compile(r'\b(GP[-\s]*\d+[A-Za-z\s,\.]*MARUTI\s*INDUSTRIAL\s*ESTATE[A-Za-z\s,\.]*)\b', re.IGNORECASE),
        re.compile(r'\b(VIHAR[A-Za-z\s,\.]*SECTOR[-\s]*\d+[A-Za-z\s,\.]*)\b', re.IGNORECASE),
        re.compile(r'\b(HARYANA\s*,?\s*INDIA)\b', re.IGNORECASE),
    ],
    "DOCTOR_NAME": [
        re.compile(r'(?:Ref\.?\s*Doctor|Ref\.?\s*By|Referring\s*(?:Doctor|Physician)|Consultant)\s*[:\-]\s*((?:Dr\.?\s+)?[A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+){1,4})', re.IGNORECASE),
        re.compile(r'(?:Ref\.?\s*Doctor|Ref\.?\s*By)\s*[:\-]\s*([A-Z]+(?:\s+[A-Z]+){1,4})\b'),
        re.compile(r'((?:Dr\.?\s+)?[A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+){1,3})\s*\n\s*(?:LAB\s*DIRECTOR|PATHOLOGIST|CONSULTANT|MEDICAL\s*DIRECTOR)', re.IGNORECASE | re.MULTILINE),
    ],
    "LAB_INFO": [
        re.compile(r'(?:Lab\.?\s*No\.?|Accession|LIS)\s*[:\-]\s*([A-Z0-9]{5,20})', re.IGNORECASE),
        re.compile(r'(?:Reported|Drawn|Collected)\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2})?)', re.IGNORECASE),
    ],
    "DOCTOR_SIGNATURE": [
        re.compile(r'([A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+)+)\s*\n\s*(?:LAB\s*DIRECTOR|PATHOLOGIST|CONSULTANT)', re.IGNORECASE | re.MULTILINE),
    ],
}

# IMPROVED PII DETECTOR CLASS
class PIIDetector:
    """Detects PII in text using regex patterns and Ollama Gemma model"""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(self.config.model in name for name in model_names):
                    logger.info(f"✓ Ollama connected, model '{self.config.model}' available")
                    return True
                else:
                    logger.warning(f"Model '{self.config.model}' not found. Available: {model_names}")
                    for name in model_names:
                        if "gemma" in name.lower():
                            self.config.model = name
                            logger.info(f"Using alternative model: {name}")
                            return True
            return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    def _is_biomarker(self, text: str) -> bool:
        """
        Context-aware biomarker detection
        Returns True if text is medical content (should NOT be redacted)
        Returns False if text is PII (should be redacted)
        """
        if not text or len(text.strip()) < 2:
            return True
    
        text_clean = text.strip()
        text_lower = text_clean.lower()
    
        # SECTION 1: MEDICAL CONTENT - Return True (DON'T redact)
        column_headers = {
            'biological reference interval', 'reference interval', 'bio. ref. interval',
            'biological interval', 'units', 'results', 'test name', 'test report status',
            'methodology', 'specimen', 'final', 'status', 'parameter',
            'specialised chemistry', 'specialized chemistry', 'hormone', 'thyroid panel',
            'lab director', 'pathologist', 'consultant', 'medical director',
            'haematology', 'hematology', 'biochemistry', 'serology', 'microbiology',
        }
        if text_lower in column_headers:
            return True
    
        # Exact medical terms
        if text_lower in MEDICAL_TERMS_EXACT or text_lower in BIOMARKERS:
            return True
    
        # Multi-word medical phrases (4+ words)
        if len(text_clean.split()) >= 4:
            medical_phrases = [
                'end of report', 'please visit', 'for related test information',
                'tests relate only', 'test requisition form', 'specimen quality is unsatisfactory',
                'turnaround time stated', 'clinical safety', 'technical integrity',
                'ordering doctor', 'assay run failure', 'asterisks are excluded',
                'laboratory is accredited', 'not valid for medico legal purposes',
                'customer care', 'investigation report', 'end of page',
            ]
            for phrase in medical_phrases:
                if phrase in text_lower:
                    return True
    
        # Long disclaimer/notes text (50+ chars with medical keywords)
        if len(text_clean) > 50 and any(keyword in text_lower for keyword in [
            'presumed', 'assayed', 'quality standards', 'technical', 'specimen',
            'turnaround', 'directory', 'requisition', 'satisfactory', 'integrity',
            'disclaimer', 'note:', 'interpretation', 'accredited', 'medico legal'
        ]):
            return True
    
        # Lab values with units
        if re.match(r'^\d+\.?\d*\s*(?:mg/dl|g/dl|mmol/l|umol/l|ng/ml|pg/ml|miu/ml|%|u/l|iu/l|x\s*10|cells/ul|meq/l|fl|pg|pmol/l|ug/dl|ug/l|mg/l|nmol/l)$', text_lower, re.IGNORECASE):
            return True

        # Reference ranges (e.g., "5.0 - 10.0", "5.0-10.0")
        if re.match(r'^\d+\.?\d*\s*[-–]\s*\d+\.?\d*$', text_lower):
            return True
    
        # Comparison operators with values (e.g., "<5", ">100", "<=10")
        if re.match(r'^[<>=]{1,2}\s*\d+\.?\d*$', text_lower):
            return True
    
        # Page numbers
        if re.match(r'^page\s*\d+', text_lower):
            return True
    
        # Single-word medical keywords
        medical_keywords = {
            'reference', 'normal', 'result', 'value', 'unit', 'range',
            'test', 'panel', 'profile', 'count', 'level', 'report',
            'serum', 'plasma', 'blood', 'final', 'interpretation',
            'interval', 'methodology', 'biological', 'units', 'results',
            'status', 'high', 'low', 'abnormal', 'negative', 'positive',
        }
        if text_lower in medical_keywords:
            return True
    
        # Multi-word phrases with medical context
        if len(text_clean.split()) >= 2:
            words_lower = [w.lower() for w in text_clean.split()]
            medical_combos = {
                'final', 'status', 'test', 'report', 'thyroid', 'panel', 'serum',
                'reference', 'interval', 'biological', 'specialised', 'specialized',
                'chemistry', 'hormone', 'end', 'please', 'visit', 'sample', 'collected',
                'liver', 'function', 'kidney', 'lipid', 'complete', 'blood', 'count',
            }
            if any(w in medical_combos for w in words_lower):
                return True
    
        # Generic medical content check (lowercase text with medical words)
        if text_lower == text_clean.lower() and any(word in text_lower for word in [
            'test', 'result', 'sample', 'specimen', 'method', 'reference',
            'interpretation', 'note', 'disclaimer', 'methodology', 'assay'
        ]):
            return True
    
        # SECTION 2: PII INDICATORS - Return False (DO redact)
        # Names with honorific prefixes (Mr./Mrs./Ms./Dr.)
        if re.match(r'^(?:mr\.|mrs\.|ms\.)\s+[a-z]', text_lower):
            return False
    
        # Doctor names with Dr. prefix followed by full name
        if re.match(r'^dr\.\s+[a-z]+(?:\s+[a-z]+)+$', text_lower):
            return False
    
        # Dates (any format with slashes or hyphens)
        if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text_clean):
            return False
    
        # Times
        if re.match(r'^\d{1,2}:\d{2}', text_clean):
            return False
    
        # ID patterns (letters + digits)
        if re.match(r'^[a-z]{1,3}\d{6,}$', text_lower) or re.match(r'^\d{4}[a-z]{2}\d{6,}$', text_lower):
            return False
    
        # Client codes
        if re.match(r'^c\d{9,}$', text_lower):
            return False
    
        # Short alphanumeric codes
        if re.match(r'^[a-z]{2}\d{6,}$', text_lower):
            return False
    
        # Age patterns
        if re.match(r'^\d{1,3}\s*(?:years?|y|yrs?)(?:\s*/\s*(?:male|female|m|f))?$', text_lower):
            return False
    
        # Gender (standalone)
        if text_lower in ['male', 'female', 'm', 'f']:
            return False
    
        # Relationship indicators (Son of, Daughter of, Wife of)
        if re.match(r'^(?:s/o|d/o|w/o)\s+', text_lower):
            return False
    
        # Indian mobile numbers
        if re.match(r'^[6-9]\d{9}$', text_clean):
            return False
    
        # Email addresses
        if re.match(r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$', text_lower):
            return False
    
        # Addresses (contains 6-digit PIN code)
        if re.search(r'\b\d{6}\b', text_clean) and len(text_clean) > 20:
            return False
    
        # Capitalized names (2+ words, title case) - but exclude known medical phrases
        if len(text_clean.split()) >= 2 and re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', text_clean):
            words_lower = [w.lower() for w in text_clean.split()]
        
            # If it contains medical combo words, it's medical content
            medical_combos = {
                'final', 'status', 'test', 'report', 'thyroid', 'panel', 'serum',
                'reference', 'interval', 'biological', 'specialised', 'specialized',
                'chemistry', 'hormone', 'liver', 'function', 'kidney', 'lipid',
                'complete', 'blood', 'count', 'end', 'page',
            }
            if any(w in medical_combos for w in words_lower):
                return True
            # Otherwise it's likely a name - should be redacted
            return False
        
        # DEFAULT: If nothing matched above, assume it's PII and should be redacted
        return False
    
    def _detect_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using improved regex patterns"""
        entities = []

        for pii_type, patterns in PII_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    if match.lastindex and match.lastindex > 0:
                        matched_text = match.group(1)
                        start = match.start(1)
                        end = match.end(1)
                    else:
                        matched_text = match.group(0)
                        start = match.start()
                        end = match.end()
                    
                    if not matched_text or len(matched_text.strip()) < 2:
                        continue
                    
                    # Log what was found for debugging
                    if pii_type == "DOCTOR_NAME":
                        logger.info(f"Regex matched DOCTOR_NAME: '{matched_text}'")
                    
                    # Check if it's medical content (biomarker) - skip if yes
                    if self._is_biomarker(matched_text):
                        logger.debug(f"Skipped biomarker: '{matched_text}'")
                        continue
                    
                    entities.append({
                        "text": matched_text.strip(),
                        "type": pii_type,
                        "start": start,
                        "end": end,
                        "confidence": 0.95
                    })
        
        return self._remove_overlapping(entities)
        
    def _detect_with_gemma(self, text: str) -> List[Dict[str, Any]]:
        prompt = f"""You are a PII redaction system for medical lab reports. Find ALL personal identifiable information.

        MUST DETECT (these are PII):
        1. Patient/Cleint names (e.g., "MR. SONU SAINI", "Mrs. KAVITA", "NIKETA", "Mr. Shrey Gupta Sharma)
        2. Age (e.g., "34 Years", "62 YEARS", "46Y")
        3. Gender/Sex (Male, Female, M, F)
        4. Dates with and without timestamps (e.g., "24-04-2024 10:37AM", "17-07-2023")
        5. Phone numbers (10 digit)
        6. Fax numbers (e.g., )
        7. Lab numbers, Accession numbers, LIS numbers
        8. Client codes (e.g., "C000132801")
        9. Doctor names (after "Ref. Doctor:", "Ref. By:")
        10. Full, multi-line physical addresses (e.g., "Flat 202, Sunshine Apts, Delhi - 110001")
        11. Location and PIN codes (6 digit numbers)

        DO NOT DETECT (these are NOT PII):
        - Test names: TSH, Haemoglobin, Glucose, WBC, RBC
        - Lab values: 9.037, 6.41, 65.5
        - Units: uIU/mL, mg/dL, %
        - Reference ranges: 0.550-4.780, 40-80
        - Headers: BIOCHEMISTRY, Test Name, Results, Units, Biological Reference Interval
        - Medical terms: serum, plasma, ultrasensitive, count, panel
        - Report text: "End of Report", "Final", "LAB DIRECTOR"

        Text:
        "{text[:2500]}"

        Return ONLY JSON array:
        [{{"text": "MR. SONU SAINI", "type": "PATIENT_NAME"}}, {{"text": "62 YEARS", "type": "AGE"}}]

        If no PII: []"""
        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.05}
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    entities = json.loads(json_match.group())
                    valid_entities = []
                    for e in entities:
                        if isinstance(e, dict) and "text" in e and "type" in e:
                            if not self._is_biomarker(e["text"]):
                                pos = text.find(e["text"])
                                if pos != -1:
                                    valid_entities.append({
                                        "text": e["text"],
                                        "type": e.get("type", "PII"),
                                        "start": pos,
                                        "end": pos + len(e["text"]),
                                        "confidence": 0.80
                                    })
                    return valid_entities
        except Exception as e:
            logger.warning(f"Gemma detection failed: {e}")
        
        return []
    
    def _remove_overlapping(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping higher confidence ones"""
        if not entities:
            return []
        
        entities = sorted(entities, key=lambda x: (x["start"], -x.get("confidence", 0), -(x["end"] - x["start"])))
        result = []
        for entity in entities:
            overlaps = False
            for existing in result:
                if not (entity["end"] <= existing["start"] or entity["start"] >= existing["end"]):
                    if entity.get("confidence", 0) > existing.get("confidence", 0):
                        result.remove(existing)
                    else:
                        overlaps = True
                    break
            if not overlaps:
                result.append(entity)
        
        return sorted(result, key=lambda x: x["start"])
    
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """Detect all PII in text using regex patterns and Gemma LLM"""
        regex_entities = self._detect_with_regex(text)
        logger.info(f"Regex detected {len(regex_entities)} PII entities")
        
        gemma_entities = self._detect_with_gemma(text)
        logger.info(f"Gemma detected {len(gemma_entities)} additional PII entities")
        
        all_entities = regex_entities.copy()
        for ge in gemma_entities:
            overlaps = False
            for re_ent in regex_entities:
                if not (ge["end"] <= re_ent["start"] or ge["start"] >= re_ent["end"]):
                    overlaps = True
                    break
            if not overlaps:
                all_entities.append(ge)
        
        final_entities = self._remove_overlapping(all_entities)
        logger.info(f"Total PII entities after merge: {len(final_entities)}")
        return final_entities

# DOCUMENT PROCESSOR
class DocumentProcessor:    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.detector = PIIDetector(config)
    
    def _image_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array"""
        if image.mode == "RGB":
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif image.mode == "RGBA":
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        else:
            return np.array(image.convert("RGB"))
    
    def _numpy_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def _get_ocr_data(self, image: np.ndarray, languages: Optional[str] = None) -> Dict[str, List]:
        """Extract OCR data with automatic multi-language support"""
        try:
            if not languages:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    osd = pytesseract.image_to_osd(gray)
                    logger.debug(f"OSD Detection: {osd}")
                except:
                    pass
                
                # English, German, Dutch, Portuguese, Spanish, Czech, French, Italian, Polish, Russian, Burmese, Thai, Vietnamese
                languages = 'eng+deu+nld+por+spa+ces+fra+ita+pol+rus+mya+tha+vie'
                logger.info(f"Using auto-detected multi-language OCR: {languages}")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Multiple threshold strategies
            thresh_adaptive = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            configs = [
                (f'--oem 3 --psm 6 -l {languages}', thresh_otsu),
                (f'--oem 3 --psm 6 -l {languages}', enhanced),
                (f'--oem 3 --psm 6 -l {languages}', thresh_adaptive),
                (f'--oem 1 --psm 3 -l {languages}', thresh_otsu),
                (f'--oem 3 --psm 11 -l {languages}', denoised),
            ]
            
            best_result = None
            best_text_count = 0

            for config, thresh in configs:
                try:
                    ocr_data = pytesseract.image_to_data(
                        thresh, output_type=Output.DICT, config=config
                    )
                    if ocr_data and 'text' in ocr_data:
                        text_count = sum(1 for t in ocr_data['text'] if t.strip())
                        if text_count > best_text_count:
                            best_text_count = text_count
                            best_result = ocr_data
                            logger.debug(f"OCR config found {text_count} text elements")
                except Exception as e:
                    logger.debug(f"OCR attempt failed: {e}")
                    continue
            
            if best_result:
                logger.info(f"Best OCR result: {best_text_count} text elements")
                return best_result
            
            # Final fallback
            return pytesseract.image_to_data(gray, output_type=Output.DICT, config=f'--oem 3 --psm 6 -l {languages}')
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {'text': [''], 'left': [0], 'top': [0], 'width': [0], 'height': [0]}
    
    def _detect_barcodes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect barcode and QR regions using visual detection"""
        return self._detect_barcode_regions_visually(image)
    
    def _detect_barcode_regions_visually(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect barcode-like regions using image processing"""
        regions = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Apply edge detection to find barcode patterns
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect barcode lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                area = w * h

                # Barcode: wide and short, reasonable size
                is_barcode = (aspect_ratio > 3 and 80 < w < width * 0.5 and 15 < h < 200 and 3000 < area < width * height * 0.15)
                
                # QR code: square, reasonable size
                is_qr = (0.7 < aspect_ratio < 1.4 and 40 < w < 400 and 40 < h < 400 and 1600 < area < 160000)
                
                if is_barcode or is_qr:
                    padding = 15
                    region = (
                        max(0, x - padding), 
                        max(0, y - padding),
                        min(width - x, w + 2 * padding), 
                        min(height - y, h + 2 * padding)
                    )
                    regions.append(region)
                    logger.info(f"Detected {'barcode' if is_barcode else 'QR'} region: x={x}, y={y}, w={w}, h={h}, aspect={aspect_ratio:.2f}")
            
            if regions:
                logger.info(f"Visual detection found {len(regions)} potential barcode/QR regions")
            else:
                logger.info("No barcode regions detected visually")
        except Exception as e:
            logger.warning(f"Visual barcode detection failed: {e}")
        
        return regions
    
    def _is_in_barcode_region(self, bbox: Tuple[int, int, int, int], barcode_regions: List[Tuple[int, int, int, int]]) -> bool:
        """Check if bbox overlaps with barcode"""
        x1, y1, w1, h1 = bbox
        for bx, by, bw, bh in barcode_regions:
            if not (x1 + w1 < bx or x1 > bx + bw or y1 + h1 < by or y1 > by + bh):
                return True
        return False
    
    def _find_text_bbox(self, entity_text: str, ocr_data: Dict[str, List]) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box with fuzzy matching"""
        words = entity_text.strip().split()
        if not words:
            return None
        
        best_match = None
        best_score = 0.0
        
        for i in range(len(ocr_data["text"])):
            ocr_word = ocr_data["text"][i].strip()
            if not ocr_word or ocr_data["width"][i] == 0:
                continue
            
            if len(words) == 1:
                similarity = self._similarity(words[0], ocr_word)
                if similarity >= self.config.ocr_similarity_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = [i]
            else:
                first_sim = self._similarity(words[0], ocr_word)
                if first_sim >= self.config.ocr_similarity_threshold:
                    matched_indices = [i]
                    word_idx = 1
                    for j in range(i + 1, min(i + len(words) * 3, len(ocr_data["text"]))):
                        next_ocr = ocr_data["text"][j].strip()
                        if not next_ocr:
                            continue
                        if word_idx < len(words):
                            sim = self._similarity(words[word_idx], next_ocr)
                            if sim >= self.config.ocr_similarity_threshold:
                                matched_indices.append(j)
                                word_idx += 1
                                if word_idx >= len(words):
                                    break
                    
                    match_ratio = len(matched_indices) / len(words)
                    if match_ratio >= 0.5 and match_ratio > best_score:
                        best_score = match_ratio
                        best_match = matched_indices
        
        if best_match:
            try:
                x_coords = [ocr_data["left"][idx] for idx in best_match if ocr_data["width"][idx] > 0]
                y_coords = [ocr_data["top"][idx] for idx in best_match if ocr_data["height"][idx] > 0]
                w_coords = [ocr_data["left"][idx] + ocr_data["width"][idx] for idx in best_match if ocr_data["width"][idx] > 0]
                h_coords = [ocr_data["top"][idx] + ocr_data["height"][idx] for idx in best_match if ocr_data["height"][idx] > 0]
                
                if x_coords and y_coords and w_coords and h_coords:
                    x_min, y_min = min(x_coords), min(y_coords)
                    x_max, y_max = max(w_coords), max(h_coords)
                    if x_max > x_min and y_max > y_min:
                        return (x_min, y_min, x_max - x_min, y_max - y_min)
            except Exception as e:
                logger.warning(f"Error calculating bbox: {e}")
        return None
    
    def _redact_image(self, image: np.ndarray, page_num: int = 1, languages: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Redact PII from image"""
        logger.info(f"=== Processing Page {page_num} ===")

        try:
            ocr_data = self._get_ocr_data(image, languages)
        except Exception as e:
            logger.error(f"Page {page_num}: OCR failed: {e}")
            return image, []
        
        full_text = " ".join(word for word in ocr_data["text"] if word.strip())
        if not full_text:
            logger.warning(f"Page {page_num}: No text found in image")
            return image, []
        
        logger.info(f"Page {page_num}: Extracted {len(full_text)} characters via OCR")
        barcode_regions = self._detect_barcodes(image)
        logger.info(f"Page {page_num}: Protected {len(barcode_regions)} barcode regions")
        pii_entities = self.detector.detect(full_text)
        logger.info(f"Page {page_num}: Detected {len(pii_entities)} PII entities to redact")
        
        # Log what PII was found
        pii_types = {}
        for entity in pii_entities:
            pii_type = entity.get('type', 'UNKNOWN')
            if pii_type not in pii_types:
                pii_types[pii_type] = []
            pii_types[pii_type].append(entity['text'][:30])  # First 30 chars
        
        for pii_type, examples in pii_types.items():
            logger.info(f"Page {page_num}: Found {len(examples)} {pii_type}: {examples[:3]}")
        
        redacted = image.copy()
        redaction_details = []
        redacted_count = 0
        skipped_count = 0
        
        for entity in pii_entities:
            bbox = self._find_text_bbox(entity["text"], ocr_data)
            if bbox:
                x, y, w, h = bbox
                if self._is_in_barcode_region(bbox, barcode_regions):
                    logger.info(f"Page {page_num}: Skipping '{entity['text'][:30]}' - in barcode region")
                    skipped_count += 1
                    continue
                
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w, h = w + 2 * padding, h + 2 * padding
                cv2.rectangle(redacted, (x, y), (x + w, y + h), self.config.redaction_color, -1)
                redaction_details.append({
                    "text": entity["text"],
                    "type": entity["type"],
                    "bbox": {"x": x, "y": y, "width": w, "height": h}
                })
                redacted_count += 1
                logger.info(f"Page {page_num}: ✓ Redacted: {entity['type']} - '{entity['text'][:30]}'")
            else:
                logger.warning(f"Page {page_num}: ✗ Could not find bbox for: {entity['type']} - '{entity['text'][:30]}'")
                skipped_count += 1
        
        logger.info(f"Page {page_num}: Redaction complete: {redacted_count} redacted, {skipped_count} skipped")
        return redacted, redaction_details
    
    def _add_overlay_text(self, image: np.ndarray, redaction_details: List[Dict], overlay_text: str) -> np.ndarray:
        """Add repeating white text over redacted areas"""
        if not overlay_text or not redaction_details:
            return image
        
        result = image.copy()
        for detail in redaction_details:
            bbox = detail["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Calculate appropriate font size based on redaction height
            font_scale = max(0.3, min(1.0, h / 40))
            font_thickness = max(1, int(font_scale * 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                overlay_text, font, font_scale, font_thickness
            )
            
            # Add some padding
            text_width += 10
            
            # Repeat text horizontally across the redacted area
            num_repeats = max(1, (w // text_width) + 1)
            
            for i in range(num_repeats):
                text_x = x + (i * text_width)
                text_y = y + (h // 2) + (text_height // 2)
                
                # Only draw if within bounds
                if text_x < x + w:
                    cv2.putText(
                        result,
                        overlay_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        self.config.text_color,
                        font_thickness,
                        cv2.LINE_AA
                    )
        return result

    def process_image(self, input_path: str, output_path: str, overlay_text: Optional[str] = None, languages: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image file"""
        logger.info(f"Processing image: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        redacted, details = self._redact_image(image)

        # Add overlay text if provided
        if overlay_text:
            redacted = self._add_overlay_text(redacted, details, overlay_text)
            logger.info(f"Added overlay text: {overlay_text}")
        
        cv2.imwrite(output_path, redacted)
        logger.info(f"Saved redacted image: {output_path}")
        return {"input": input_path, "output": output_path, "pages": 1, "pii_detected": details}
    
    def process_pdf(self, input_path: str, output_path: str, overlay_text: Optional[str] = None, languages: Optional[str] = None) -> Dict[str, Any]:
        """Process a PDF file"""
        logger.info(f"Processing PDF: {input_path}")
        try:
            images = pdf2image.convert_from_path(input_path, dpi=self.config.dpi)
        except Exception as e1:
            poppler_path = os.environ.get("POPPLER_PATH")
            if poppler_path and os.path.exists(poppler_path):
                logger.info(f"Retrying with POPPLER_PATH: {poppler_path}")
                try:
                    images = pdf2image.convert_from_path(input_path, dpi=self.config.dpi, poppler_path=poppler_path)
                except Exception as e2:
                    logger.error(f"PDF conversion error with provided path: {e2}")
                    raise ValueError(f"Could not convert PDF: {e2}")
            else:
                logger.error(f"PDF conversion error: {e1}. Ensure Poppler is installed and in PATH.")
                raise ValueError(f"Could not convert PDF: {e1}")
        
        if not images:
            raise ValueError("No pages found in PDF")
        
        logger.info(f"PDF has {len(images)} pages")
        all_details = []
        processed_images = []
        
        for page_num, pil_image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            img_array = self._image_to_numpy(pil_image)
            redacted, page_details = self._redact_image(img_array)
            
            # Add overlay text if provided
            if overlay_text:
                redacted = self._add_overlay_text(redacted, page_details, overlay_text)
            
            for detail in page_details:
                detail["page"] = page_num
            all_details.extend(page_details)
            processed_images.append(self._numpy_to_pil(redacted))
        
        image_bytes = []
        for img in processed_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            image_bytes.append(buf)
        
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(image_bytes))
        
        logger.info(f"Saved redacted PDF: {output_path}")
        return {"input": input_path, "output": output_path, "pages": len(images), "pii_detected": all_details}
    
    def process(self, input_path: str, output_path: str, overlay_text: Optional[str] = None, languages: Optional[str] = None) -> Dict[str, Any]:
        """Process any supported document type"""
        ext = Path(input_path).suffix.lower()
        if ext == ".pdf":
            return self.process_pdf(input_path, output_path, overlay_text)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return self.process_image(input_path, output_path, overlay_text)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

# FASTAPI APPLICATION
app = FastAPI(
    title="Medical Document PII Redaction API",
    description="API for detecting and redacting PII from medical documents",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor: Optional[DocumentProcessor] = None

@app.on_event("startup")
async def startup():
    global processor
    logger.info("Starting PII Redaction API v2.0...")
    processor = DocumentProcessor(Config())
    logger.info("API ready!")

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Medical Document PII Redaction API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/redact": "POST - Upload document for PII redaction",
            "/health": "GET - Health check",
            "/docs": "GET - Swagger UI documentation"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "processor_ready": processor is not None
    }

@app.post("/redact", tags=["Redaction"])
async def redact_document(
    file: UploadFile = File(...),
    overlay_text: Optional[str] = Form(None, description="Optional text to overlay on redacted areas")
):
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    
    session_id = uuid4().hex[:8]
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    output_filename = f"redacted_{session_id}_{file.filename}"
    output_path = OUTPUT_DIR / output_filename
    
    try:
        logger.info(f"[{session_id}] Received: {file.filename}")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"[{session_id}] Processing with auto-detected OCR languages...")
        if overlay_text:
            logger.info(f"[{session_id}] Using overlay text: {overlay_text}")
        
        result = processor.process(input_path, str(output_path), overlay_text, None)
        
        pii_count = len(result.get("pii_detected", []))
        logger.info(f"[{session_id}] Complete! Redacted {pii_count} PII items across {result['pages']} page(s)")
        
        shutil.rmtree(temp_dir)
        
        media_type = "application/pdf" if ext == ".pdf" else f"image/{ext[1:]}"
        return FileResponse(path=str(output_path), filename=output_filename, media_type=media_type)
        
    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)