"""Minimal rule layer. Not a moderation system."""
from .tokenizer import tokenize


class Safety:
    def __init__(self, banned_terms=None, required_terms=None,
                 max_output_tokens=200, max_input_tokens=2000):
        self.banned_terms = [t.lower() for t in (banned_terms or [])]
        self.required_terms = [t.lower() for t in (required_terms or [])]
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens

    def check_input(self, text):
        toks = tokenize(text)
        if len(toks) > self.max_input_tokens:
            return False, "input_too_long"
        low = text.lower()
        for term in self.banned_terms:
            if term and term in low:
                return False, f"banned_term:{term}"
        return True, "ok"

    def filter_output(self, text):
        if not text:
            return ""
        low = text.lower()
        for term in self.banned_terms:
            if term and term in low:
                return "[response withheld by safety rule]"
        toks = tokenize(text)
        if len(toks) > self.max_output_tokens:
            text = " ".join(text.split()[:self.max_output_tokens])
        # Reject obviously degenerate output (same token repeated many times)
        if len(toks) >= 8 and len(set(toks)) <= 2:
            return ""
        return text.strip()

    def satisfies_required(self, text):
        if not self.required_terms:
            return True
        low = text.lower()
        return all(t in low for t in self.required_terms)
