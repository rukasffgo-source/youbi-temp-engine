"""Per-session short-term conversation memory."""
from collections import defaultdict, deque


class Memory:
    def __init__(self, max_messages=20):
        self.max_messages = max_messages
        self._sessions = defaultdict(lambda: deque(maxlen=max_messages))

    def add(self, session_id, role, content):
        self._sessions[session_id].append({"role": role, "content": content})

    def history(self, session_id):
        return list(self._sessions[session_id])

    def clear(self, session_id=None):
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)
