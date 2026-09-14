"""Youbi HTTP API. Flask is the only third-party dep.

The API contract is stable: /api/generate, /api/chat, /api/ingest,
/api/reset, /api/status. Swap TemporaryYoubiAI for RealYoubiAI and
nothing else changes.
"""
import os
from flask import Flask, request, jsonify, render_template
from engine import TemporaryYoubiAI

DATA_DIR = os.environ.get("YOUBI_DATA_DIR", "data")
app = Flask(__name__)
engine = TemporaryYoubiAI(data_dir=DATA_DIR)


def _params(body):
    return {
        "session_id": body.get("session_id", "default"),
        "temperature": float(body.get("temperature", 0.7)),
        "max_tokens": int(body.get("max_tokens", 120)),
        "top_k": int(body.get("top_k", 40)),
    }


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True, silent=True) or {}
    prompt = body.get("prompt", "")
    return jsonify(engine.generate(prompt, **_params(body)))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/api/chat", methods=["POST"])
def chat():
    # Alias for /api/generate; kept because frontends often use /chat.
    body = request.get_json(force=True, silent=True) or {}
    prompt = body.get("prompt") or body.get("message", "")
    return jsonify(engine.generate(prompt, **_params(body)))


@app.route("/api/ingest", methods=["POST"])
def ingest():
    body = request.get_json(force=True, silent=True) or {}
    if "path" in body:
        info = engine.ingest_file(body["path"])
        return jsonify({"ok": True, **info})
    text = body.get("text", "")
    source = body.get("source", "<memory>")
    info = engine.ingest(text, source=source)
    return jsonify({"ok": True, **info})


@app.route("/api/reset", methods=["POST"])
def reset():
    body = request.get_json(force=True, silent=True) or {}
    return jsonify(engine.reset(body.get("session_id")))


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify(engine.status())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
