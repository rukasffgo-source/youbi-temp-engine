"""Backend abstraction. Real Youbi will subclass YoubiAI and drop in here."""

class YoubiAI:
    name = "abstract-youbi"

    def generate(self, prompt, session_id=None, **kwargs):
        raise NotImplementedError

    def ingest(self, text, source="<memory>", **kwargs):
        raise NotImplementedError

    def reset(self, session_id=None):
        raise NotImplementedError

    def status(self):
        return {
            "engine": self.name,
            "type": "abstract",
            "learned_neural_model": False,
            "ready_for_model_swap": True,
        }
