"""
Language utilities for multilingual text processing.
Provides language detection, tokenization, and preprocessing for English and Chinese.
"""

import re
from typing import List, Set, Tuple, Optional
import pandas as pd


# Embedding models for different language modes
EMBEDDING_MODELS = {
    'english': 'all-MiniLM-L6-v2',
    'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2'
}


def detect_language(text: str) -> str:
    """
    Detect if text is Chinese or English based on character ratio.

    Args:
        text: Input text string

    Returns:
        'zh' for Chinese, 'en' for English
    """
    if not text or pd.isna(text):
        return 'en'
    text = str(text)
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ratio = chinese_chars / max(len(text), 1)
    return 'zh' if ratio > 0.3 else 'en'


def get_jieba():
    """Lazy load jieba to avoid import overhead when not needed."""
    try:
        import jieba
        return jieba
    except ImportError:
        raise ImportError(
            "jieba not installed. Install with: pip install jieba"
        )


def tokenize_text(text: str, language: str = None) -> List[str]:
    """
    Tokenize text based on language.

    Args:
        text: Input text string
        language: 'en', 'zh', or None (auto-detect)

    Returns:
        List of tokens
    """
    if not text or pd.isna(text):
        return []

    text = str(text)

    if language is None:
        language = detect_language(text)

    if language == 'zh':
        jieba = get_jieba()
        return list(jieba.cut(text))

    # English: simple whitespace tokenization
    return text.split()


def tokenize_text_for_tfidf(text: str) -> List[str]:
    """
    Tokenize text for TF-IDF vectorizer (auto-detects language).
    This function is used as a custom tokenizer for sklearn's TfidfVectorizer.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    return tokenize_text(text, language=None)


def preprocess_text_multilingual(text: str) -> str:
    """
    Preprocess text for multilingual use.

    Args:
        text: Input text string

    Returns:
        Preprocessed text string
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Lowercase only affects English, harmless for Chinese
    return text.lower()


def extract_words(text: str, language: str = None) -> Set[str]:
    """
    Extract words from text based on language.

    Args:
        text: Input text string
        language: 'en', 'zh', or None (auto-detect)

    Returns:
        Set of words
    """
    tokens = tokenize_text(text, language)
    return set(tokens)


def get_bigrams_multilingual(text: str, language: str = None) -> Set[Tuple[str, str]]:
    """
    Extract bigrams based on language.
    For Chinese, uses character bigrams for better coverage.
    For English, uses word bigrams.

    Args:
        text: Input text string
        language: 'en', 'zh', or None (auto-detect)

    Returns:
        Set of bigram tuples
    """
    if not text or pd.isna(text):
        return set()

    text = str(text)

    if language is None:
        language = detect_language(text)

    if language == 'zh':
        # Character bigrams for Chinese
        chars = [c for c in text if not c.isspace()]
        if len(chars) < 2:
            return set()
        return set(zip(chars[:-1], chars[1:]))
    else:
        # Word bigrams for English
        words = text.split()
        if len(words) < 2:
            return set()
        return set(zip(words[:-1], words[1:]))


def get_first_n_tokens(text: str, n: int = 3, language: str = None) -> List[str]:
    """
    Get first N tokens from text.

    Args:
        text: Input text string
        n: Number of tokens to return
        language: 'en', 'zh', or None (auto-detect)

    Returns:
        List of first n tokens
    """
    tokens = tokenize_text(text, language)
    return tokens[:n] if len(tokens) >= n else tokens


def extract_capitalized_words_multilingual(text: str, language: str = None) -> Set[str]:
    """
    Extract capitalized words (entity-like) from text.
    For Chinese, returns empty set as Chinese has no capitalization.

    Args:
        text: Input text string
        language: 'en', 'zh', or None (auto-detect)

    Returns:
        Set of capitalized words (lowercased)
    """
    if not text or pd.isna(text):
        return set()

    text = str(text)

    if language is None:
        language = detect_language(text)

    if language == 'zh':
        # Chinese has no capitalization concept
        return set()

    # English: extract capitalized words
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    return set(w.lower() for w in words)


def extract_numbers(text: str) -> Set[str]:
    """
    Extract all numbers from text (language-agnostic).

    Args:
        text: Input text string

    Returns:
        Set of number strings
    """
    if not text or pd.isna(text):
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))


def get_embedding_model_name(language_mode: str) -> str:
    """
    Get the appropriate embedding model name for the language mode.

    Args:
        language_mode: 'english' or 'multilingual'

    Returns:
        Model name string
    """
    return EMBEDDING_MODELS.get(language_mode, EMBEDDING_MODELS['english'])
