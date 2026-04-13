"""
Japanese Tokenizer: SudachiPy-based tokenizer for Japanese text processing.

Fix 9 (P1): Standard whitespace tokenization doesn't work for Japanese.
This module provides morphological analysis using SudachiPy (or a fallback
regex-based tokenizer when SudachiPy is not installed).

Usage:
    tokenizer = JapaneseTokenizer()
    tokens = tokenizer.tokenize("東京都の天気は晴れです")
    # With SudachiPy: ['東京都', 'の', '天気', 'は', '晴れ', 'です']
    # Fallback:       ['東京都の天気は晴れです'] (character-based)
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Attempt to import SudachiPy; fall back gracefully
_SUDACHI_AVAILABLE = False
try:
    from sudachipy import tokenizer as sudachi_tokenizer
    from sudachipy import dictionary as sudachi_dictionary
    _SUDACHI_AVAILABLE = True
except ImportError:
    logger.info("SudachiPy not installed. Using regex fallback tokenizer. "
                "Install with: pip install sudachipy sudachidict_core")


class JapaneseTokenizer:
    """
    SudachiPy tokenizer for Japanese text processing.
    
    Falls back to a regex-based character tokenizer when SudachiPy
    is not available, ensuring the pipeline never crashes on import.
    
    Modes (SudachiPy):
        A: Short unit (most granular, e.g., 東京 + 都)
        B: Medium unit (balanced)
        C: Long unit (named entities kept together, e.g., 東京都)
    """
    
    def __init__(self, mode: str = 'C'):
        """
        Args:
            mode: SudachiPy split mode ('A', 'B', or 'C')
        """
        self._sudachi = None
        self._mode = None
        
        if _SUDACHI_AVAILABLE:
            try:
                dic = sudachi_dictionary.Dictionary()
                modes = {
                    'A': sudachi_tokenizer.Tokenizer.SplitMode.A,
                    'B': sudachi_tokenizer.Tokenizer.SplitMode.B,
                    'C': sudachi_tokenizer.Tokenizer.SplitMode.C,
                }
                self._mode = modes.get(mode, modes['C'])
                self._sudachi = dic.create()
                logger.info(f"SudachiPy tokenizer initialized (mode={mode})")
            except Exception as e:
                logger.warning(f"SudachiPy init failed: {e}. Using fallback.")
                self._sudachi = None
        
        # Regex patterns for fallback tokenizer
        self._ja_pattern = re.compile(
            r'[\u3040-\u309F]+|'    # Hiragana
            r'[\u30A0-\u30FF]+|'    # Katakana
            r'[\u4E00-\u9FFF]+|'    # CJK Unified Ideographs (Kanji)
            r'[\uFF66-\uFF9F]+|'    # Half-width Katakana
            r'[a-zA-Z0-9]+|'       # ASCII alphanumeric
            r'[^\s]'                # Any other non-whitespace
        )
    
    @property
    def backend(self) -> str:
        """Return the active tokenizer backend name"""
        return 'sudachi' if self._sudachi else 'regex_fallback'
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Japanese text into morphemes.
        
        Args:
            text: Input text (Japanese or mixed)
            
        Returns:
            List of surface-form tokens
        """
        if not text or not text.strip():
            return []
        
        if self._sudachi:
            morphemes = self._sudachi.tokenize(text, self._mode)
            return [m.surface() for m in morphemes]
        else:
            return self._fallback_tokenize(text)
    
    def tokenize_normalized(self, text: str) -> List[str]:
        """
        Tokenize and normalize to dictionary form.
        
        Dictionary form converts inflected words to their base form:
        e.g., 走った → 走る
        
        Falls back to surface form when SudachiPy is unavailable.
        """
        if not text or not text.strip():
            return []
        
        if self._sudachi:
            morphemes = self._sudachi.tokenize(text, self._mode)
            return [m.dictionary_form() for m in morphemes]
        else:
            return self._fallback_tokenize(text)
    
    def tokenize_with_pos(self, text: str) -> List[dict]:
        """
        Tokenize with part-of-speech information.
        
        Returns list of {surface, pos, dictionary_form} dicts.
        Only available with SudachiPy backend.
        """
        if not text or not text.strip():
            return []
        
        if self._sudachi:
            morphemes = self._sudachi.tokenize(text, self._mode)
            return [
                {
                    'surface': m.surface(),
                    'pos': m.part_of_speech(),
                    'dictionary_form': m.dictionary_form(),
                    'reading': m.reading_form(),
                }
                for m in morphemes
            ]
        else:
            tokens = self._fallback_tokenize(text)
            return [
                {
                    'surface': t,
                    'pos': ['Unknown'],
                    'dictionary_form': t,
                    'reading': t,
                }
                for t in tokens
            ]
    
    def _fallback_tokenize(self, text: str) -> List[str]:
        """Regex-based fallback tokenizer for Japanese text"""
        return self._ja_pattern.findall(text)
    
    def __call__(self, text: str) -> List[str]:
        """Callable interface for sklearn-compatible tokenizer parameter"""
        return self.tokenize(text)


def create_tfidf_tokenizer() -> JapaneseTokenizer:
    """
    Factory function for creating a tokenizer compatible with
    sklearn's TfidfVectorizer.
    
    Usage:
        from src.japanese_tokenizer import create_tfidf_tokenizer
        tokenizer = create_tfidf_tokenizer()
        
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None  # Disable default pattern
        )
    """
    return JapaneseTokenizer(mode='C')
