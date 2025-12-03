# utils.py - Enhanced version with better text cleaning and processing
import re
import numpy as np
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory
import unicodedata

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Initialize spell checkers for multiple languages
spell_en = SpellChecker(language='en')
spell_ar = SpellChecker(language=None)  # Arabic spell checking is limited

# Unicode ranges covering common Arabic blocks
ARABIC_RANGES = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"

def detect_content_type(text):
    """Enhanced content type detection"""
    if not text or len(text.strip()) < 5:
        return "unknown"
    
    # Count different character types
    arabic_chars = len(re.findall(rf'[{ARABIC_RANGES}]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    digits = len(re.findall(r'\d', text))
    
    total_meaningful = arabic_chars + latin_chars + digits
    
    if total_meaningful == 0:
        return "unknown"
    
    arabic_ratio = arabic_chars / total_meaningful
    latin_ratio = latin_chars / total_meaningful
    
    if arabic_ratio > 0.6:
        return "arabic"
    elif latin_ratio > 0.6:
        return "english"
    elif arabic_ratio > 0.2 and latin_ratio > 0.2:
        return "mixed"
    else:
        return "multilingual"

def normalize_arabic(text: str) -> str:
    """Enhanced Arabic text normalization"""
    if not text:
        return ""
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
    
    # Normalize different forms of alef
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ٱ', 'ا')
    
    # Normalize ya and hamza variants
    text = text.replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي')
    
    # Remove tatweel (kashida)
    text = text.replace('ـ', '')
    
    # Normalize spaces around Arabic text
    text = re.sub(r'([' + ARABIC_RANGES + r'])\s+([' + ARABIC_RANGES + r'])', r'\1 \2', text)
    
    return text

def remove_toc_artifacts(text: str) -> str:
    """Remove table of contents and document structure artifacts"""
    if not text:
        return ""
    
    # Patterns for common TOC and document artifacts
    patterns_to_remove = [
        # Table of contents patterns
        r'^\s*(TABLE\s+OF\s+)?CONTENTS?\s*$',
        r'^\s*ABSTRACT\s*\.{3,}.*$',
        r'^\s*INTRODUCTION\s*\.{3,}.*$',
        r'^\s*CONCLUSION\s*\.{3,}.*$',
        r'^\s*REFERENCES?\s*\.{3,}.*$',
        r'^\s*BIBLIOGRAPHY\s*\.{3,}.*$',
        
        # Page numbers and references
        r'^\s*Page\s+\d+\s*$',
        r'^\s*\d+\s*$',  # Lone numbers
        r'^\s*[ivxlcdm]+\s*$',  # Roman numerals
        
        # Excessive punctuation
        r'\.{5,}',  # 5 or more dots
        r'-{5,}',   # 5 or more dashes
        r'_{5,}',   # 5 or more underscores
        r'={5,}',   # 5 or more equals
        
        # Header/footer artifacts
        r'^\s*\d+\s*$',  # Just page numbers
        r'^\s*Chapter\s+\d+\s*$',  # Chapter headers
        r'^\s*Section\s+[\d.]+\s*$',  # Section headers
        
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        keep_line = True
        
        for pattern in patterns_to_remove:
            if re.match(pattern, line.strip(), re.IGNORECASE | re.MULTILINE):
                keep_line = False
                break
        
        if keep_line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_spacing_and_punctuation(text: str) -> str:
    """Clean up spacing and punctuation issues"""
    if not text:
        return ""
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])(?=[^\s])', r'\1 ', text)  # Add space after punctuation
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix multiple newlines (keep max 2)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove trailing/leading whitespace on each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def remove_noise_patterns(text: str) -> str:
    """Remove common OCR noise patterns and junk"""
    if not text:
        return ""
    
    noise_patterns = [
        # Single char / symbol lines
        r'^\s*[^\w\u0600-\u06FF\s]\s*$',
        
        # Repeated Latin letters like "aaaa", "zzzz"
        r'\b([a-zA-Z])\1{3,}\b',
        
        # Repeated Arabic letters like "ههه", "دددد"
        rf'([{ARABIC_RANGES}])\1{{3,}}',
        
        # Long sequences of numbers or codes
        r'\b\d{6,}\b',  # 6+ digit numbers
        r'(?:\d+\s+){4,}',  # long sequences of numbers separated by spaces
        
        # Weird punctuation sequences
        r'[^\w\s\u0600-\u06FF]{4,}',
        
        # Lines that look like OCR gibberish (too many symbols/digits, not enough words)
        r'^[\d\sOIZS]{6,}$',  # numbers / OCR confusions
        
        # Lines with mixed junk (e.g. "09 01 88 ض 685 20-1 206")
        r'^(?:[A-Za-z0-9\-\s]{8,})$',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        is_noise = False
        for pattern in noise_patterns:
            if re.match(pattern, line):
                is_noise = True
                break
        
        # Extra filter: drop lines with < 30% letters
        letters = len(re.findall(r'[a-zA-Z' + ARABIC_RANGES + ']', line))
        ratio = letters / max(1, len(line))
        if ratio < 0.3 and len(line) > 12:  # too few letters for its length
            is_noise = True
        
        if not is_noise:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
def is_meaningful_text(text: str) -> bool:
    """Decide if text is meaningful enough to keep"""
    if not text or len(text.strip()) < 20:
        return False

    # Count letters vs total
    letters = len(re.findall(r'[a-zA-Z' + ARABIC_RANGES + ']', text))
    ratio = letters / max(1, len(text))
    if ratio < 0.4:  # less than 40% letters = noise
        return False

    # Try language detection
    try:
        lang = detect(text)
        if lang not in ["ar", "en"]:
            return False
    except:
        return False

    return True


def clean_ocr_noise(text: str, spellcheck: bool = False, lang: str = "auto") -> str:
    """Enhanced OCR cleaning with strict noise removal"""
    if not text:
        return ""

    # Remove obvious junk
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Apply existing cleaning steps
    text = remove_toc_artifacts(text)
    text = remove_noise_patterns(text)
    text = clean_spacing_and_punctuation(text)

    # Normalize Arabic
    content_type = detect_content_type(text)
    if content_type in ['arabic', 'mixed', 'multilingual']:
        text = normalize_arabic(text)

    # Remove repeated sequences (common OCR duplication)
    words = text.split()
    if len(words) > 10:
        for seq_len in [4, 3, 2]:
            i = 0
            while i < len(words) - seq_len * 2:
                if words[i:i+seq_len] == words[i+seq_len:i+2*seq_len]:
                    words = words[:i+seq_len] + words[i+2*seq_len:]
                else:
                    i += 1
        text = " ".join(words)

    # Final spacing fix
    text = clean_spacing_and_punctuation(text)

    # Last guard: drop if not meaningful
    if not is_meaningful_text(text):
        return ""

    return text.strip()
# def clean_ocr_noise(text: str, spellcheck: bool = False, lang: str = "auto") -> str:
#     """Enhanced text cleaning with better Arabic/English support"""
#     if not text:
#         return ""
    
#     # Detect content type
#     content_type = detect_content_type(text)
#     print(f"Detected content type: {content_type}")
    
#     # Remove URLs and email addresses
#     text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    
#     # Remove HTML tags if present
#     text = re.sub(r'<[^>]+>', ' ', text)
    
#     # Remove table of contents artifacts
#     text = remove_toc_artifacts(text)
    
#     # Remove noise patterns
#     text = remove_noise_patterns(text)
    
#     # Normalize Arabic text if present
#     if content_type in ['arabic', 'mixed', 'multilingual']:
#         text = normalize_arabic(text)
    
#     # Clean spacing and punctuation
#     text = clean_spacing_and_punctuation(text)
    
#     # Remove repeated word/phrase sequences (common OCR error)
#     words = text.split()
#     if len(words) > 10:  # Only for longer texts
#         # Remove repeated sequences of 2-4 words
#         for seq_len in [4, 3, 2]:
#             i = 0
#             while i < len(words) - seq_len * 2:
#                 current_seq = words[i:i+seq_len]
#                 next_seq = words[i+seq_len:i+seq_len*2]
#                 if current_seq == next_seq:
#                     # Remove the repeated sequence
#                     words = words[:i+seq_len] + words[i+seq_len*2:]
#                 else:
#                     i += 1
#         text = ' '.join(words)
    
#     # Language detection for spellcheck
#     detected_lang = lang
#     if lang == "auto":
#         try:
#             detected_lang = detect(text)
#         except:
#             detected_lang = "en" if content_type == "english" else "ar"
    
#     # Filter out very short "words" that are likely noise
#     words = text.split()
#     filtered_words = []
    
#     for word in words:
#         # Keep word if:
#         # - It's long enough
#         # - It contains meaningful characters
#         # - It's not just punctuation or numbers
#         if len(word) >= 2:
#             if re.search(r'[a-zA-Z\u0600-\u06FF]', word):  # Contains letters
#                 filtered_words.append(word)
#             elif len(word) >= 3 and re.search(r'\d', word):  # Numbers with 3+ digits
#                 filtered_words.append(word)
    
#     text = ' '.join(filtered_words)
    
#     # Spell checking (limited for Arabic)
#     if spellcheck and content_type in ["english", "mixed"]:
#         corrected_words = []
#         for word in text.split():
#             if re.match(r'^[a-zA-Z]+$', word) and len(word) > 2:
#                 # Only correct English words
#                 corrected = spell_en.correction(word.lower())
#                 if corrected and corrected != word.lower():
#                     # Only use correction if it's significantly different or the original isn't a valid English word
#                     if word.lower() not in spell_en:
#                         corrected_words.append(corrected)
#                     else:
#                         corrected_words.append(word)
#                 else:
#                     corrected_words.append(word)
#             else:
#                 corrected_words.append(word)
#         text = ' '.join(corrected_words)
    
#     # Final cleanup
#     text = clean_spacing_and_punctuation(text)
    
#     # Remove very short segments that are likely noise
#     # Final cleanup: drop meaningless segments
#     segments = text.split('\n')
#     meaningful_segments = []
    
#     for seg in segments:
#         seg = seg.strip()
#         if not seg:
#             continue
        
#         # Drop very short noise
#         if len(seg) < 8:
#             continue
        
#         # Drop segments with too many numbers/symbols
#         letters = len(re.findall(r'[a-zA-Z' + ARABIC_RANGES + ']', seg))
#         if letters / max(1, len(seg)) < 0.4:
#             continue
        
#         meaningful_segments.append(seg)
    
#     return '\n'.join(meaningful_segments).strip()



def clean_text(text: str, spellcheck: bool = False, lang: str = "en") -> str:
    """Clean and optionally spellcheck text."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s\u0600-\u06FF.,;:!?()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if spellcheck and lang == "en":
        spell = SpellChecker()
        corrected = []
        for word in text.split():
            corrected.append(spell.correction(word) or word)
        text = " ".join(corrected)

    return text



def embed_texts(embedder, texts, batch_size=32) -> np.ndarray:
    """Enhanced embedding with better error handling and batching"""
    if not texts:
        return np.empty((0, 384), dtype="float32")  # Default dimension for MiniLM
    
    # Filter and clean texts
    valid_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            # Truncate very long texts to avoid embedding issues
            cleaned = text.strip()[:8000]  # Max 8000 chars
            if len(cleaned) > 10:  # Minimum length
                valid_texts.append(cleaned)
    
    if not valid_texts:
        return np.empty((0, 384), dtype="float32")
    
    print(f"Embedding {len(valid_texts)} text chunks...")
    
    all_embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        try:
            batch_embeddings = embedder.encode(batch)
            all_embeddings.append(batch_embeddings)
            print(f"Embedded batch {i//batch_size + 1}/{(len(valid_texts)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Create zero embeddings for failed batch
            dummy_emb = np.zeros((len(batch), 384), dtype="float32")
            all_embeddings.append(dummy_emb)
    
    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        return np.empty((0, 384), dtype="float32")

def chunk_text_intelligently(text: str, max_tokens: int = 300, overlap_tokens: int = 50):
    """Intelligent text chunking that respects sentence and paragraph boundaries"""
    if not text or not text.strip():
        return []
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for paragraph in paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r'[.!?]+\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                
                # Start new chunk with overlap
                if overlap_tokens > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_words = ' '.join(current_chunk).split()[-overlap_tokens:]
                    current_chunk = [' '.join(overlap_words)]
                    current_tokens = len(overlap_words)
                else:
                    current_chunk = []
                    current_tokens = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    # Filter out very short chunks
    filtered_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 5]
    
    return filtered_chunks