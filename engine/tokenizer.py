"""Unicode-aware tokenizer. No English assumptions.

Splits on Unicode whitespace and Unicode punctuation, but keeps
apostrophes/hyphens inside words so 'ntibaza-gutya' stays one token.
Works for Kinyarwanda, English, French, Hindi, etc.
"""
import re
import unicodedata

# Anything that is NOT a letter, digit, or an in-word apostrophe/hyphen.
# \w in Python 3 re is Unicode-aware by default, so this covers
# Kinyarwanda/French/Hindi letters too.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """NFKC normalize + collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    """Return list of tokens (words/punctuation-free chunks)."""
    return _TOKEN_RE.findall(normalize(text).lower())


def words(text: str):
    """Word-only tokens (for keyword extraction)."""
    return _WORD_RE.findall(normalize(text).lower())
