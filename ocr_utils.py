# ocr_utils.py
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from pathlib import Path
import os
import docx2txt
import cv2
import numpy as np
from langdetect import detect, DetectorFactory
import easyocr
import torch

# =====================
# Environment
# =====================
POPPLER_PATH = r"C:\poppler\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

DetectorFactory.seed = 0  # make langdetect deterministic

# =====================
# Initialize EasyOCR once
# =====================
try:
    easyocr_reader = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available())
except:
    easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)


# =====================
# Helpers
# =====================
def detect_language_safe(text: str) -> str:
    """Safe language detection with fallback."""
    try:
        lang = detect(text)
        if lang == "ar":
            return "arabic"
        elif lang == "en":
            return "english"
        else:
            return "multilingual"
    except:
        return "unknown"


def deskew_image(gray):
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(pil_img, upscale=None):
    """Preprocess for better OCR results."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    try:
        thresh = deskew_image(thresh)
    except:
        pass
    if upscale:
        thresh = cv2.resize(
            thresh, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR
        )
    return Image.fromarray(thresh)


# =====================
# OCR
# =====================
def ocr_with_dual_pass(img):
    """OCR with separate Arabic & English runs, fallback to EasyOCR."""
    img_proc = preprocess_image(img)

    best_text = ""
    best_lang = "unknown"

    # --- Arabic only ---
    try:
        ar_text = pytesseract.image_to_string(
            img_proc, lang="ara", config="--psm 11"
        ).strip()
        if ar_text:
            ar_lang = detect_language_safe(ar_text)
            if ar_lang == "arabic":
                return ar_text
            if len(ar_text) > len(best_text):
                best_text, best_lang = ar_text, ar_lang
    except:
        pass

    # --- English only ---
    try:
        en_text = pytesseract.image_to_string(
            img_proc, lang="eng", config="--psm 11"
        ).strip()
        if en_text:
            en_lang = detect_language_safe(en_text)
            if en_lang == "english":
                return en_text
            if len(en_text) > len(best_text):
                best_text, best_lang = en_text, en_lang
    except:
        pass

    # --- EasyOCR fallback ---
    try:
        results = easyocr_reader.readtext(
            np.array(img_proc), detail=0, paragraph=True
        )
        text = "\n".join(results).strip()
        if text:
            lang = detect_language_safe(text)
            if lang != "unknown":
                return text
    except:
        pass

    return best_text


def _ocr_image(img):
    """OCR wrapper"""
    return ocr_with_dual_pass(img)


# =====================
# Extraction
# =====================
def extract_text(path: str):
    """Extract text from PDF, images, DOCX, TXT with OCR fallback."""
    path = str(path)
    suffix = Path(path).suffix.lower()
    pages_text = []

    if suffix == ".pdf":
        # Try vector PDF extraction first
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text.strip())
        except:
            pass

        # OCR fallback if no text
        if not pages_text:
            images = convert_from_path(
                path, poppler_path=POPPLER_PATH if POPPLER_PATH else None
            )
            for i, img in enumerate(images, start=1):
                text = _ocr_image(img)
                if text.strip():
                    lang = detect_language_safe(text)
                    print(f"[OCR] Page {i}: Detected {lang}")
                    pages_text.append(text.strip())

    elif suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        img = Image.open(path)
        text = _ocr_image(img)
        if text.strip():
            lang = detect_language_safe(text)
            print(f"[OCR] Image: Detected {lang}")
            pages_text.append(text.strip())

    elif suffix == ".docx":
        text = docx2txt.process(path)
        if text.strip():
            pages_text.append(text.strip())

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            pages_text.append(text.strip())

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return pages_text


# =====================
# Deduplication
# =====================
def deduplicate_pages(pages: list[str]) -> list[str]:
    """Remove duplicate lines across pages."""
    cleaned = []
    seen_lines = set()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        new_lines = [l for l in lines if l not in seen_lines]
        seen_lines.update(new_lines)
        cleaned.append(" ".join(new_lines))
    return cleaned
