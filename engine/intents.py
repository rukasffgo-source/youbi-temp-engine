"""Intent classification via weighted keyword scoring. No neural nets.

Each intent has a list of (pattern, weight) pairs. Patterns are plain
substrings (case-insensitive, Unicode-normalized). Score = sum of
matched weights, normalized into a confidence in [0, 1].
"""
from .tokenizer import normalize

# Order matters only for ties; higher-scoring intent wins.
INTENT_RULES = {
    "greeting": [
        ("hello", 2), ("hi ", 2), ("hey", 2), ("muraho", 2), ("mwaramutse", 2),
        ("mwiriwe", 2), ("bonjour", 2), ("salut", 2), ("namaste", 2),
        ("good morning", 3), ("good evening", 3), ("how are you", 2),
    ],
    "goodbye": [
        ("bye", 2), ("goodbye", 3), ("see you", 2), ("murabeho", 3),
        ("au revoir", 3), ("farewell", 3), ("later", 1),
    ],
    "help": [
        ("what can you do", 4), ("help", 2), ("how do you work", 3),
        ("what are you", 3), ("who are you", 3), ("your capabilities", 3),
    ],
    "summarization": [
        ("summarize", 4), ("summary", 3), ("tl;dr", 4), ("shorten", 2),
        ("in short", 3), ("brief", 1),
    ],
    "rewriting": [
        ("rewrite", 4), ("rephrase", 4), ("paraphrase", 4),
        ("say it differently", 4), ("reword", 3),
    ],
    "translation": [
        ("translate", 5), ("in french", 2), ("in english", 2),
        ("in kinyarwanda", 3), ("in hindi", 3), ("translation", 3),
    ],
    "coding": [
        ("code", 3), ("function", 2), ("python", 3), ("javascript", 3),
        ("bug", 2), ("debug", 3), ("implement", 2), ("algorithm", 2),
        ("sql", 2), ("regex", 2),
    ],
    "explanation": [
        ("explain", 4), ("what is", 2), ("what are", 2), ("why", 2),
        ("how does", 2), ("define", 3), ("meaning of", 3),
    ],
    "brainstorming": [
        ("brainstorm", 4), ("ideas", 3), ("suggest", 2), ("name ideas", 4),
        ("list of", 2), ("options for", 2),
    ],
    "creative_writing": [
        ("write a story", 5), ("poem", 4), ("song", 3), ("screenplay", 4),
        ("fiction", 3), ("paragraph about", 3), ("compose", 3),
    ],
    "question": [
        ("?", 2), ("how ", 1), ("what ", 1), ("when ", 1), ("where ", 1),
        ("who ", 1), ("which ", 1),
    ],
    "casual": [
        ("lol", 2), ("haha", 2), ("cool", 1), ("nice", 1), ("thanks", 2),
        ("thank you", 3), ("murakoze", 2),
    ],
}

# Confidence below this -> "unknown"
UNKNOWN_THRESHOLD = 0.30


def detect_intent(prompt: str):
    """Return {'intent': str, 'confidence': float, 'scores': dict}."""
    text = " " + normalize(prompt).lower() + " "
    scores = {}
    for intent, rules in INTENT_RULES.items():
        score = 0
        for pat, weight in rules:
            if pat in text:
                score += weight
        if score:
            scores[intent] = score

    if not scores:
        return {"intent": "unknown", "confidence": 0.0, "scores": {}}

    best = max(scores, key=scores.get)
    raw = scores[best]
    # Normalize: 1 point -> 0.33, 3 points -> 0.6, 6+ points -> ~0.8+
    confidence = raw / (raw + 2.0)
    if confidence < UNKNOWN_THRESHOLD:
        return {"intent": "unknown", "confidence": confidence, "scores": scores}
    return {"intent": best, "confidence": round(confidence, 3), "scores": scores}
