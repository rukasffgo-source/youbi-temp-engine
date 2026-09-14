"""TemporaryYoubiAI — hybrid intent + retrieval + n-gram. Replaceable."""
import os
from .base import YoubiAI
from .intents import detect_intent
from .retrieval import DocumentStore
from .ngram import NGramModel
from .memory import Memory
from .safety import Safety
from .tokenizer import normalize, tokenize

# High-quality deterministic templates for common intents.
# These win over n-gram generation when the intent is confident.
TEMPLATES = {
    "greeting": [
        "Hey! I'm Youbi. What do you want to create?",
        "Hi — I'm Youbi. What are we working on today?",
        "Hello! Youbi here. Give me something to chew on.",
    ],
    "goodbye": [
        "See you. Come back when you have something to build.",
        "Goodbye! I'll be here.",
    ],
    "help": [
        "I'm a small temporary Youbi engine. I can:\n"
        "- answer questions using documents you've ingested,\n"
        "- rewrite, summarize, and explain short text,\n"
        "- brainstorm lists and names,\n"
        "- hold a short conversation with memory.\n"
        "I'm not a real LLM — I'll tell you when I don't know something.",
    ],
    "summarization": [
        "Paste the text you want summarized and I'll condense it.",
    ],
    "rewriting": [
        "Send me the text you want rewritten and I'll rephrase it.",
    ],
    "translation": [
        "I don't do real translation. I can only match templates and "
        "retrieved text. A real model will replace this soon.",
    ],
    "coding": [
        "I can hold small code snippets and explain them lexically, but "
        "I can't write reliable code. Use me as a scratchpad, not a compiler.",
    ],
    "creative_writing": [
        "I can produce short, rough n-gram continuations, but they won't "
        "read like a real story. Feed me examples to steer the style.",
    ],
    "brainstorming": [
        "Tell me the topic and I'll list angles from what I've seen.",
    ],
    "casual": [
        "Cool. What do you want to do?",
        "Got it. What's next?",
    ],
    "explanation": [
        "I'll try. If I don't have the info ingested, I'll say so.",
    ],
}


class TemporaryYoubiAI(YoubiAI):
    name = "temporary-youbi"

    def __init__(self, data_dir="data", ngram_order=3, max_memory=20,
                 seed_texts=None, safety=None):
        self.data_dir = data_dir
        self.store = DocumentStore()
        self.ngram = NGramModel(order=ngram_order)
        self.memory = Memory(max_messages=max_memory)
        self.safety = safety or Safety()
        self._bootstrap(seed_texts or [])
        self._load_if_present()

    # ---- bootstrapping ----
    def _bootstrap(self, seed_texts):
        # A tiny built-in corpus so the n-gram model isn't empty at boot.
        builtin = (
            "Youbi is a small temporary assistant. "
            "Youbi can help you create, explain, rewrite and explore ideas. "
            "Youbi is honest about what it can and cannot do. "
            "Youbi runs on ordinary hardware without a neural network. "
            "Youbi will be replaced by a real model later. "
            "Muraho, nitwa Youbi. Youbi ifasha gukora no gusobanura ibintu. "
        )
        self.ngram.train(builtin)
        self.store.add_document(builtin, source="<builtin>")
        for t in seed_texts:
            self.ngram.train(t)
            self.store.add_document(t, source="<seed>")

    def _load_if_present(self):
        model_path = os.path.join(self.data_dir, "youbi_model.json")
        docs_path = os.path.join(self.data_dir, "youbi_documents.json")
        if os.path.exists(model_path):
            try:
                self.ngram = NGramModel.load(model_path)
            except Exception:
                pass
        if os.path.exists(docs_path):
            try:
                self.store.load(docs_path)
            except Exception:
                pass

    # ---- public API ----
    def generate(self, prompt, session_id=None, temperature=0.7,
                 max_tokens=120, top_k=40, **kwargs):
        prompt = normalize(prompt or "")
        session_id = session_id or "default"

        # 1. Safety check on input
        ok, reason = self.safety.check_input(prompt)
        if not ok:
            return self._envelope(
                text=f"[blocked: {reason}]", intent="blocked",
                confidence=1.0, retrieved=[], prompt=prompt)

        # 2. Empty input
        if not prompt:
            return self._envelope(
                text="Say something and I'll do my best.",
                intent="unknown", confidence=0.0, retrieved=[], prompt=prompt)

        # 3. Intent
        intent_info = detect_intent(prompt)
        intent = intent_info["intent"]
        confidence = intent_info["confidence"]

        # 4. Retrieval (always attempted; used if it scores well)
        retrieved = self.store.search(prompt, top_k=3)
        top_retrieval_score = retrieved[0][0] if retrieved else 0.0

        # 5. Response strategy
        text = ""
        used = []
        if intent in TEMPLATES and confidence >= 0.35:
            text = self._pick_template(intent)
        elif top_retrieval_score >= 0.20:
            text = self._from_retrieval(prompt, retrieved)
            used = [{"source": c["source"], "score": round(s, 3)}
                    for s, c in retrieved]
        else:
            text = self._from_ngram(prompt, temperature, max_tokens, top_k)
            if not text or len(text.split()) < 3:
                text = self._fallback(prompt, intent)

        # 6. Format + safety
        text = self.safety.filter_output(text)
        if not text:
            text = self._fallback(prompt, intent)

        # 7. Memory
        self.memory.add(session_id, "user", prompt)
        self.memory.add(session_id, "assistant", text)

        return self._envelope(text, intent, confidence, used, prompt,
                              session_id=session_id)

    def ingest(self, text, source="<memory>", **kwargs):
        text = normalize(text or "")
        if not text:
            raise ValueError("empty document")
        self.store.add_document(text, source=source)
        self.ngram.train(text)
        return {"ingested_chars": len(text), "source": source,
                "chunks": len(self.store.chunks)}

    def ingest_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.store.load_file(path)
        # Also train the n-gram model on the raw text where possible
        if path.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                self.ngram.train(f.read())
        return {"source": path}

    def reset(self, session_id=None):
        self.memory.clear(session_id)
        return {"cleared": session_id or "all"}

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.ngram.save(os.path.join(self.data_dir, "youbi_model.json"))
        self.store.save(os.path.join(self.data_dir, "youbi_documents.json"))
        return {"saved": True, "dir": self.data_dir}

    def status(self):
        return {
            "engine": self.name,
            "type": "hybrid-n-gram-retrieval",
            "learned_neural_model": False,
            "ready_for_model_swap": True,
            "ngram_order": self.ngram.order,
            "documents": len(self.store.chunks),
            "vocab_size": len(self.ngram.vocab),
            "trained": self.ngram.trained,
        }

    # ---- internals ----
    def _envelope(self, text, intent, confidence, retrieved, prompt,
                  session_id="default"):
        return {
            "text": text,
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "engine": self.name,
            "retrieved_sources": retrieved,
            "session_id": session_id,
            "prompt_tokens": len(tokenize(prompt)),
        }

    def _pick_template(self, intent):
        import random
        return random.choice(TEMPLATES[intent])

    def _from_retrieval(self, prompt, retrieved):
        # Stitch together the best snippets with a short prefix.
        head = "Here's what I found in my knowledge base:\n"
        pieces = []
        for score, chunk in retrieved[:2]:
            pieces.append(f"- ({chunk['source']}) {chunk['text'][:400]}")
        return head + "\n".join(pieces)

    def _from_ngram(self, prompt, temperature, max_tokens, top_k):
        # Cap output; n-gram output degrades fast.
        cap = min(int(max_tokens or 60), 60)
        return self.ngram.generate(
            prompt, max_tokens=cap, temperature=temperature, top_k=top_k)

    def _fallback(self, prompt, intent):
        return (
            "I'm a small temporary engine, so I can't answer that well yet. "
            "Try ingesting a document on the topic, or ask me to "
            "summarize/rewrite/brainstorm something specific. "
            "(intent detected: {})".format(intent)
        )
