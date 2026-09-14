"""N-gram language model. Markov-style, no neural nets."""
import json
import math
import random
from collections import defaultdict, Counter
from .tokenizer import tokenize


class NGramModel:
    """Order-N n-gram model with temperature, top-K, and repetition penalty."""

    def __init__(self, order=3):
        assert order >= 1
        self.order = order
        # context tuple -> Counter(next_token -> freq)
        self.transitions = defaultdict(Counter)
        self.vocab = Counter()
        self.starts = []          # observed (order-1)-token prefixes
        self.trained = False

    # ---- training ----
    def train(self, text):
        tokens = tokenize(text)
        if len(tokens) < self.order:
            return
        self.vocab.update(tokens)
        n = self.order
        for i in range(len(tokens) - n + 1):
            ctx = tuple(tokens[i:i + n - 1])
            nxt = tokens[i + n - 1]
            self.transitions[ctx][nxt] += 1
            if i == 0 or (i > 0 and i < len(tokens) - n + 1):
                self.starts.append(ctx)
        self.trained = True

    def train_many(self, texts):
        for t in texts:
            self.train(t)

    # ---- sampling ----
    def _sample(self, counter, temperature, top_k, banned):
        # Build (token, prob) after filtering
        items = [(tok, c) for tok, c in counter.items() if tok not in banned]
        if not items:
            items = list(counter.items())
        if not items:
            return None

        # top-K by frequency
        if top_k and top_k > 0 and len(items) > top_k:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:top_k]

        # temperature: 0 -> argmax, large -> uniform
        if temperature <= 1e-6:
            return max(items, key=lambda x: x[1])[0]
        weights = [c ** (1.0 / temperature) for _, c in items]
        total = sum(weights)
        if total <= 0:
            return random.choice(items)[0]
        r = random.random() * total
        acc = 0.0
        for (tok, _), w in zip(items, weights):
            acc += w
            if acc >= r:
                return tok
        return items[-1][0]

    def generate(self, prompt, max_tokens=60, temperature=0.7, top_k=40,
                 repetition_penalty=1.15):
        """Generate continuation tokens from prompt context."""
        if not self.trained:
            return ""

        tokens = tokenize(prompt)
        ctx_len = self.order - 1

        # Seed context: last (order-1) tokens of prompt, or a stored start.
        if ctx_len == 0:
            ctx = ()
        elif len(tokens) >= ctx_len:
            ctx = tuple(tokens[-ctx_len:])
        elif self.starts:
            ctx = random.choice(self.starts)
        else:
            return ""

        out = []
        recent = Counter()
        for _ in range(max_tokens):
            counter = self.transitions.get(ctx)
            if not counter:
                # back off: try shorter contexts
                for back in range(1, ctx_len + 1):
                    shorter = ctx[back:] if back <= len(ctx) else ()
                    counter = self.transitions.get(shorter)
                    if counter:
                        break
                if not counter:
                    break

            # Repetition penalty: down-weight recently used tokens
            adjusted = Counter()
            for tok, c in counter.items():
                penalty = repetition_penalty ** recent[tok]
                adjusted[tok] = max(1e-9, c / penalty)

            nxt = self._sample(adjusted, temperature, top_k, banned=set())
            if nxt is None:
                break
            out.append(nxt)
            recent[nxt] += 1

            ctx = (ctx + (nxt,))[-(ctx_len):] if ctx_len else ()

        return " ".join(out)

    # ---- persistence ----
    def to_dict(self):
        return {
            "order": self.order,
            "transitions": {",".join(k): dict(v)
                            for k, v in self.transitions.items()},
            "vocab": dict(self.vocab),
            "starts": [list(s) for s in self.starts[:5000]],
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = cls(order=data.get("order", 3))
        for k, v in data.get("transitions", {}).items():
            ctx = tuple(k.split(",")) if k else ()
            m.transitions[ctx] = Counter(v)
        m.vocab = Counter(data.get("vocab", {}))
        m.starts = [tuple(s) for s in data.get("starts", [])]
        m.trained = bool(m.transitions)
        return m
