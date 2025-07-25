import re

def clean_text(text: str) -> str:
    # Basic cleanup: remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def truncate_text(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def summarize_text(text: str, max_sentences: int = 3) -> str:
    # Very basic summarizer: take first N sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:max_sentences])
