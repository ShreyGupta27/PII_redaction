# 🛡️ PII Redaction

## Overview

This project provides a **high-accuracy PII (Personally Identifiable Information) redaction system** for medical documents using:

* OCR (Tesseract)
* Rule-based regex detection
* LLM-assisted detection (Ollama + Gemma)
* Image processing (OpenCV)

It supports **PDFs and images**, detects sensitive data, and redacts it while preserving medical content.

---

## 🚀 Key Features

### 🔍 Intelligent PII Detection

* Hybrid detection approach:

  * Regex-based detection (high precision)
  * LLM-based detection via Ollama (context-aware)
* Detects:

  * Patient names
  * Age / Gender / DOB
  * Phone numbers, emails
  * IDs (Lab No, Patient ID, LIS, etc.)
  * Addresses
  * Doctor names

### 🧠 Medical Context Awareness

* Prevents redaction of:

  * Biomarkers (e.g., Hemoglobin, TSH)
  * Lab values and units
  * Medical terminology
* Uses:

  * Predefined medical vocabulary
  * Custom biomarker whitelist (`biomarkers_example.json`)

### 🖼️ OCR + Image Processing

* Multi-language OCR support
* Adaptive preprocessing:

  * Denoising
  * Thresholding
  * Contrast enhancement
* Fuzzy matching for accurate bounding boxes

### 📄 Document Support

* PDF (multi-page)
* Images (`.jpg`, `.jpeg`, `.png`)

### 🔒 Smart Redaction

* Black-box redaction of detected regions
* Optional overlay text (e.g., "REDACTED")
* Avoids redacting:

  * Barcodes / QR codes

---

## ⚙️ Installation

### Install Dependencies

```bash
pip install fastapi uvicorn opencv-python numpy pillow pytesseract pdf2image img2pdf requests
```

---

## 🔧 System Dependencies

### ✅ Tesseract OCR

Install Tesseract:

* Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Linux:

```bash
sudo apt install tesseract-ocr
```

### ✅ Poppler (for PDF processing)

Required for `pdf2image`:

* Windows: Download Poppler and set `POPPLER_PATH`
* Linux:

```bash
sudo apt install poppler-utils
```

---

## 🤖 Optional: Ollama (LLM Support)

This project uses **Ollama + Gemma model** for enhanced PII detection.

### Install Ollama

```bash
https://ollama.ai
```

### Pull Model

```bash
ollama pull gemma2:2b-instruct-q8_0
```

### Default Config

```python
ollama_url = "http://localhost:11434"
model = "gemma2:2b-instruct-q8_0"
```

---

## 🔄 Processing Pipeline

```
Document → OCR → Text Extraction
         → PII Detection (Regex + LLM)
         → Bounding Box Matching
         → Image Redaction
         → Output File
```

---

## 🧪 Detection Logic

### 1. Regex-Based Detection

* High-confidence structured patterns
* Covers IDs, phone numbers, dates, etc.

### 2. LLM-Based Detection

* Uses Gemma model via Ollama
* Detects context-based PII missed by regex

### 3. Biomarker Filtering

* Prevents false positives using:

  * Medical vocabulary
  * Domain-specific heuristics

---

## 📊 Output

* Returns **redacted file** (PDF/image)
* Logs include:

  * PII types detected
  * Count of redactions
  * Processing details

---

## ⚠️ Limitations

* OCR accuracy depends on image quality
* Complex layouts may affect bounding box precision
* No GPU acceleration by default

---

## 📌 Use Cases

* Medical report anonymization
* Healthcare data compliance (HIPAA-like workflows)
* Data sharing for research
* Document sanitization pipelines

---

## 🧾 License

Copyright (c) 2026 Shrey Gupta. All rights reserved.

Unauthorized copying, modification, or distribution of this software, via any medium, is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 👨‍💻 Author

Shrey Gupta (ShreyGupta27)
