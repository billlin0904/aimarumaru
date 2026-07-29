import threading
from typing import Optional

import opencc


_converter = opencc.OpenCC("s2tw.json")
_converter_lock = threading.Lock()


def is_chinese_language(language: Optional[str]) -> bool:
    if language is None:
        return True
    normalized = language.strip().lower().replace("_", "-")
    return normalized == "zh" or normalized.startswith("zh-")


def to_traditional_chinese(text: str, language: Optional[str] = None) -> str:
    if not text or not is_chinese_language(language):
        return text
    with _converter_lock:
        return _converter.convert(text)
