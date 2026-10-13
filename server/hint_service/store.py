"""hint_service — the data plane of the scheduling side.

Standalone Python package: zero third-party dependencies (stdlib HTTP); can
be lifted into a dedicated repository or scheduling service as-is.

Contract with the agent-hints plugin:

  GET  /v1/hints?prefix_id={hash}&user={u}&model={m}
       -> 200 {"session": {...}, "prediction": {...}, ...}   # partial AgentHint
  POST /v1/hints
       body {"entries": [{"prefix_id": "...", "hint": {...partial AgentHint...}}]}
       -> 200 {"upserted": n}                                # register/update
  POST /agent-status                                         # lifecycle status sink
       body {"session": {"session_id": ...}, "agent_status": {...}}
       -> 200 {}
  GET  /healthz -> 200

Run: python -m hint_service --port 9200 --store hints.json
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


class HintStore:
    """JSON-file backed store: prefix_id -> partial AgentHint. Thread-safe, atomic writes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._status: dict[str, dict] = {}
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
                self._entries = data.get("entries", {})
                self._status = data.get("status", {})
            except (json.JSONDecodeError, OSError):
                # empty/corrupt file: start fresh (self-heals on the next write)
                self._entries = {}
                self._status = {}

    def _flush(self) -> None:
        if not self.path:
            return
        payload = json.dumps({"entries": self._entries, "status": self._status},
                             ensure_ascii=False)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)

    def upsert(self, entries: list[dict]) -> int:
        count = 0
        with self._lock:
            for entry in entries:
                prefix_id = str(entry.get("prefix_id") or "").strip()
                hint = entry.get("hint")
                if not prefix_id or not isinstance(hint, dict):
                    continue
                existing = self._entries.get(prefix_id, {})
                merged = dict(existing)
                _deep_merge(merged, hint)
                self._entries[prefix_id] = merged
                count += 1
            self._flush()
        return count

    def query(self, prefix_id: str) -> dict:
        with self._lock:
            return dict(self._entries.get(prefix_id, {}))

    def record_status(self, session_id: str, agent_status: dict) -> None:
        with self._lock:
            self._status[session_id] = agent_status
            self._flush()

    def status_snapshot(self) -> dict:
        with self._lock:
            return dict(self._status)


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
