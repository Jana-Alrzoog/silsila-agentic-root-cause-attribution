"""Ollama LLM Client for local inference with qwen2.5:3b."""

import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
from silsila.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, SILSILA_DISABLE_LLM


# Arabic numeral mapping
ARABIC_INDIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

ARABIC_WORD_NUMBERS = {
    'يوم': 1, 'يوما': 1, 'يومان': 2, 'يومين': 2,
    'واحد': 1, 'واحدة': 1,
    'اثنين': 2, 'اثنان': 2,
    'ثلاثة': 3, 'ثلاث': 3,
    'أربعة': 4, 'اربعة': 4, 'أربع': 4, 'اربع': 4,
    'خمسة': 5, 'خمس': 5,
    'ستة': 6, 'ست': 6,
    'سبعة': 7, 'سبع': 7,
    'ثمانية': 8, 'ثماني': 8, 'ثمان': 8,
    'تسعة': 9, 'تسع': 9,
    'عشرة': 10, 'عشر': 10,
    'أسبوع': 7, 'اسبوع': 7, 'أسبوعين': 14, 'اسبوعين': 14,
}


def normalize_arabic_numerals(text: str) -> str:
    """Replaces Arabic-Indic digits (٠-٩) with ASCII digits (0-9)."""
    for ar_digit, en_digit in ARABIC_INDIC_DIGITS.items():
        text = text.replace(ar_digit, en_digit)
    return text


def extract_arabic_number_from_text(text: str) -> Optional[int]:
    """Fallback extractor for days quantity in Arabic leave requests.
    Prioritizes duration context (e.g. 'لمدة 5 أيام') over date numbers (e.g. 'من 20 نوفمبر').
    """
    normalized = normalize_arabic_numerals(text)

    # 1. Duration keyword followed by number (and optional day unit)
    # e.g., 'لمدة 5 أيام', 'لمدة 5', 'فترة 3 أيام'
    m_dur = re.search(r'(?:لمدة|فترة|مدتها)\s*(\d+)\s*(?:يوم|ايام|أيام|day|days)?', normalized)
    if m_dur:
        try:
            return int(m_dur.group(1))
        except ValueError:
            pass

    # 1b. Duration keyword followed by word number (e.g., 'لمدة أسبوع', 'لمدة يومين')
    m_dur_word = re.search(r'(?:لمدة|فترة|مدتها)\s*([\w]+)', normalized)
    if m_dur_word and m_dur_word.group(1) in ARABIC_WORD_NUMBERS:
        return ARABIC_WORD_NUMBERS[m_dur_word.group(1)]

    # 2. Number explicitly followed by day unit: e.g., '5 أيام', '10 يوم'
    m_unit = re.search(r'(\d+)\s*(?:يوم|ايام|أيام|day|days)\b', normalized)
    if m_unit:
        try:
            return int(m_unit.group(1))
        except ValueError:
            pass

    # 3. Dedicated multi-day words like أسبوعين, أسبوع, يومين
    for word, val in [('أسبوعين', 14), ('اسبوعين', 14), ('أسبوع', 7), ('اسبوع', 7), ('يومين', 2), ('يومان', 2)]:
        if word in text:
            return val

    # 4. Word number followed by day unit: e.g., 'خمسة أيام'
    for word, val in ARABIC_WORD_NUMBERS.items():
        if re.search(rf'\b{word}\s*(?:يوم|ايام|أيام|day|days)\b', text):
            return val

    # 5. Clean out dates (e.g. 'من 20 نوفمبر', 'في 15 أكتوبر') to prevent false positive duration extraction
    clean_text = re.sub(
        r'(?:من|في|بتاريخ|إلى|الى)\s*\d+\s*(?:يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر)?',
        '',
        normalized
    )
    m_clean = re.search(r'(\d+)', clean_text)
    if m_clean:
        try:
            return int(m_clean.group(1))
        except ValueError:
            pass

    # 6. Fallback to general textual word numbers
    words = re.findall(r'[\w]+', text)
    for word in words:
        if word in ARABIC_WORD_NUMBERS:
            return ARABIC_WORD_NUMBERS[word]

    return None


def extract_leave_type_from_text(text: str) -> Optional[str]:
    """Extracts expected leave type category from raw Arabic request text."""
    lower = text.lower()
    if "بدون راتب" in text or "غير مدفوع" in text or "غير مدفوعة" in text or "unpaid" in lower:
        return "unpaid"
    if "مرضية" in text or "مرضي" in text or "مريض" in text or "sick" in lower:
        return "sick"
    if "طارئة" in text or "طارئ" in text or "اضطرارية" in text or "اضطراري" in text or "emergency" in lower:
        return "emergency"
    if "سنوية" in text or "سنوي" in text or "اعتيادية" in text or "اعتيادي" in text or "annual" in lower:
        return "annual"
    return None


class OllamaClient:
    """Client for local Ollama server running qwen2.5:3b."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        """Check if local Ollama server is running and accessible."""
        if SILSILA_DISABLE_LLM:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format_json: bool = False,
        temperature: float = 0.0,
        timeout: int = OLLAMA_TIMEOUT,
        num_predict: int = 250,
    ) -> str:
        """Generate response from local Ollama model."""
        if SILSILA_DISABLE_LLM:
            raise RuntimeError("LLM inference is disabled via SILSILA_DISABLE_LLM=1")

        endpoint = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_json:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to communicate with Ollama at {endpoint}: {e}")
