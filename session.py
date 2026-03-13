"""In-memory session manager with TTL for conversation context."""

import time
from datetime import datetime

SESSION_TTL = 7200  # 2 hours


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, user_id: str) -> list:
        """Return conversation history for a user.

        Creates a new session if none exists or the existing one expired.
        """
        now = time.time()
        session = self._sessions.get(user_id)

        if session and (now - session["last_access"]) < SESSION_TTL:
            session["last_access"] = now
            return session["history"]

        # Session expired — save last session timestamp to memory
        if session:
            self._save_session_timestamp(user_id)

        # New or expired session
        self._sessions[user_id] = {
            "history": [],
            "last_access": now,
        }
        return self._sessions[user_id]["history"]

    def _save_session_timestamp(self, user_id: str | None = None):
        """Save the current time as last_session_timestamp in memory."""
        try:
            from memory import load_memory, save_memory
            memory = load_memory(user_id)
            memory["last_session_timestamp"] = datetime.now().isoformat()
            save_memory(memory, user_id)
        except Exception:
            pass  # Don't break session flow if memory save fails

    def cleanup(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            uid
            for uid, s in self._sessions.items()
            if (now - s["last_access"]) >= SESSION_TTL
        ]
        for uid in expired:
            del self._sessions[uid]
