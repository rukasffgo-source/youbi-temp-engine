"""Lexical TF-IDF-style document retrieval. Not semantic."""
import json
import math
import os
from collections import Counter
from .tokenizer import words


def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping word-chunks. Language-agnostic."""
    toks = text.split()
    if not toks:
        return []
    if len(toks) <= chunk_size:
        return [text.strip()]
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(toks), step):
        chunk = " ".join(toks[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(toks):
            break
    return chunks


class DocumentStore:
    """Stores chunks + inverted index. Pure Python, no external deps."""

    def __init__(self):
        self.chunks = []        # list of dicts {text, source, keywords}
        self._df = Counter()    # document frequency per token
        self._tf = []           # list of Counter(token -> freq) per chunk

    # ---- ingestion ----
    def add_document(self, text, source="<memory>", keywords=None):
        for chunk in chunk_text(text):
            toks = words(chunk)
            if not toks:
                continue
            tf = Counter(toks)
            meta = {
                "text": chunk,
                "source": source,
                "keywords": keywords or [w for w, _ in tf.most_common(8)],
            }
            self.chunks.append(meta)
            self._tf.append(tf)
            for tok in tf:
                self._df[tok] += 1

    def load_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8") as f:
                self.add_document(f.read(), source=os.path.basename(path))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "text" in item:
                        self.add_document(str(item["text"]),
                                          source=item.get("source", f"{path}#{i}"))
                    else:
                        self.add_document(str(item), source=f"{path}#{i}")
            else:
                self.add_document(json.dumps(data, ensure_ascii=False),
                                  source=os.path.basename(path))
        elif ext == ".csv":
            import csv
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    self.add_document(" ".join(row), source=f"{path}#row{i}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ---- search ----
    def search(self, query, top_k=3):
        """Cosine-ish TF-IDF ranking. Returns list of (score, chunk_dict)."""
        if not self.chunks or not query:
            return []
        q_tokens = words(query)
        if not q_tokens:
            return []
        q_tf = Counter(q_tokens)
        n = len(self.chunks)

        def idf(tok):
            return math.log((1 + n) / (1 + self._df.get(tok, 0))) + 1.0

        q_vec = {t: q_tf[t] * idf(t) for t in q_tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored = []
        for idx, tf in enumerate(self._tf):
            # quick skip: no shared token
            shared = q_tf.keys() & tf.keys()
            if not shared:
                continue
            dot = 0.0
            d_norm_sq = 0.0
            for tok, freq in tf.items():
                w = freq * idf(tok)
                d_norm_sq += w * w
                if tok in q_vec:
                    dot += w * q_vec[tok]
            d_norm = math.sqrt(d_norm_sq) or 1.0
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, self.chunks[idx]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    # ---- persistence ----
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks}, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = []
        self._df = Counter()
        self._tf = []
        for meta in data.get("chunks", []):
            self.add_document(meta["text"], meta.get("source", "?"),
                              meta.get("keywords"))
