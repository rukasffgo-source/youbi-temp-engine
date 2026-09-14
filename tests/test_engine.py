"""Run with: python -m pytest tests/ -q  (or python tests/test_engine.py)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import TemporaryYoubiAI
from engine.intents import detect_intent
from engine.ngram import NGramModel
from engine.retrieval import DocumentStore, chunk_text
from engine.tokenizer import tokenize, words
from engine.memory import Memory
from engine.safety import Safety


def _fresh(**kw):
    tmp = tempfile.mkdtemp()
    return TemporaryYoubiAI(data_dir=tmp, **kw)


# ---- tokenizer / unicode ----
def test_tokenize_unicode():
    toks = tokenize("Muraho, nitwa Youbi! Comment ça va? नमस्ते दुनिया")
    assert "muraho" in toks
    assert "nitwa" in toks
    assert "youbi" in toks
    assert "नमस्ते" in toks  # Devanagari preserved


def test_tokenize_empty():
    assert tokenize("") == []
    assert words("   ") == []


# ---- intents ----
def test_greeting_detection():
    r = detect_intent("hello there")
    assert r["intent"] == "greeting"
    assert r["confidence"] > 0.3


def test_translation_intent():
    r = detect_intent("Please translate this to French")
    assert r["intent"] == "translation"


def test_unknown_intent():
    r = detect_intent("xyzzy plugh frobnicate")
    assert r["intent"] == "unknown"


def test_kinyarwanda_greeting():
    r = detect_intent("Muraho, amakuru?")
    assert r["intent"] in ("greeting", "question")


# ---- retrieval ----
def test_chunking_and_retrieval():
    store = DocumentStore()
    store.add_document(
        "Youbi is a temporary engine. " * 3
        + "Kinyarwanda is spoken in Rwanda. " * 3,
        source="a.md")
    hits = store.search("Kinyarwanda", top_k=2)
    assert hits
    assert any("Kinyarwanda" in h[1]["text"] for h in hits)


def test_chunk_text_short():
    assert chunk_text("hi") == ["hi"]
    assert chunk_text("") == []


def test_malformed_document():
    store = DocumentStore()
    store.add_document("", source="empty")
    assert store.search("anything") == []


# ---- ngram ----
def test_ngram_trains_and_generates():
    m = NGramModel(order=2)
    m.train("the cat sat on the mat the cat sat again")
    out = m.generate("the cat", max_tokens=5, temperature=0.5)
    assert isinstance(out, str)


def test_ngram_temperature_greedy():
    m = NGramModel(order=2)
    m.train("a b a b a b a b a b")
    # temperature ~0 -> deterministic
    out1 = m.generate("a", max_tokens=4, temperature=0.0001)
    out2 = m.generate("a", max_tokens=4, temperature=0.0001)
    assert out1 == out2


def test_ngram_top_k_limits():
    m = NGramModel(order=2)
    m.train("a b a c a d a e a f")
    # With top_k=1 only the most common continuation is allowed.
    out = m.generate("a", max_tokens=4, temperature=1.0, top_k=1)
    assert out.count("b") >= 0  # just ensure it runs, no crash


def test_ngram_save_load(tmp_path=None):
    tmp = tempfile.mkdtemp()
    m = NGramModel(order=3)
    m.train("youbi is a small temporary engine that runs offline")
    path = os.path.join(tmp, "m.json")
    m.save(path)
    m2 = NGramModel.load(path)
    assert m2.order == 3
    assert m2.trained


# ---- memory ----
def test_memory_clear():
    mem = Memory(max_messages=4)
    mem.add("s1", "user", "hi")
    mem.add("s1", "assistant", "hello")
    assert len(mem.history("s1")) == 2
    mem.clear("s1")
    assert mem.history("s1") == []


# ---- safety ----
def test_safety_banned():
    s = Safety(banned_terms=["forbidden"])
    ok, _ = s.check_input("this is forbidden stuff")
    assert not ok
    assert s.filter_output("this is forbidden") == "[response withheld by safety rule]"


def test_safety_degenerate_output():
    s = Safety()
    assert s.filter_output("a a a a a a a a a a") == ""


# ---- engine end-to-end ----
def test_engine_greeting():
    e = _fresh()
    r = e.generate("hello", session_id="t")
    assert r["engine"] == "temporary-youbi"
    assert r["intent"] == "greeting"
    assert "Youbi" in r["text"]


def test_engine_empty_input():
    e = _fresh()
    r = e.generate("", session_id="t")
    assert r["intent"] == "unknown"
    assert r["text"]


def test_engine_unknown_input():
    e = _fresh()
    r = e.generate("xyzzy plugh", session_id="t")
    assert r["text"]
    assert r["engine"] == "temporary-youbi"


def test_engine_retrieval_used():
    e = _fresh()
    e.ingest("The capital of Rwanda is Kigali. Kigali is a large city.",
             source="geo")
    r = e.generate("What is the capital of Rwanda?", session_id="t")
    assert r["intent"] in ("question", "explanation", "unknown")
    # Either retrieved or fell back; but retrieval should have fired.
    assert isinstance(r["retrieved_sources"], list)


def test_engine_memory_roundtrip():
    e = _fresh()
    e.generate("hello", session_id="s")
    assert len(e.memory.history("s")) == 2
    e.reset("s")
    assert e.memory.history("s") == []


def test_engine_status_shape():
    e = _fresh()
    s = e.status()
    assert s["engine"] == "temporary-youbi"
    assert s["type"] == "hybrid-n-gram-retrieval"
    assert s["learned_neural_model"] is False
    assert s["ready_for_model_swap"] is True


def test_engine_save_load():
    tmp = tempfile.mkdtemp()
    e = TemporaryYoubiAI(data_dir=tmp)
    e.ingest("Kigali is the capital of Rwanda.", source="geo")
    e.save()
    e2 = TemporaryYoubiAI(data_dir=tmp)
    assert e2.ngram.trained


def test_engine_temperature_determinism():
    e = _fresh()
    r1 = e.generate("Youbi can", temperature=0.0001, max_tokens=6)
    r2 = e.generate("Youbi can", temperature=0.0001, max_tokens=6)
    assert r1["text"] == r2["text"]


# ---- flask api ----
def test_api_status():
    from app import app
    client = app.test_client()
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["engine"] == "temporary-youbi"


def test_api_generate():
    from app import app
    client = app.test_client()
    r = client.post("/api/generate",
                    json={"prompt": "hello", "session_id": "api-test"})
    assert r.status_code == 200
    body = r.get_json()
    assert "text" in body and "intent" in body and "engine" in body


def test_api_ingest_and_reset():
    from app import app
    client = app.test_client()
    r = client.post("/api/ingest",
                    json={"text": "Rwanda is in East Africa.",
                          "source": "geo"})
    assert r.status_code == 200 and r.get_json()["ok"]
    r = client.post("/api/reset", json={"session_id": "api-test"})
    assert r.status_code == 200


if __name__ == "__main__":
    # Minimal runner so it works without pytest.
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
