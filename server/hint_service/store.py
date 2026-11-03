"""hint_service — the data plane of the scheduling side.

Standalone Python package: zero third-party dependencies (stdlib HTTP); can
be lifted into a dedicated repository or scheduling service as-is.

Contract with the agent-hints plugin:

  GET  /v1/hints?prefix_id={hash}&user={u}&model={m}
       -> 200 {"session": {...}, "prediction": {...}, ...}   # partial AgentHint
  POST /v1/hints
       body {"entries": [{"prefix_id": "...", "hint": {...partial AgentHint...}}]}
       -> 200 {"upserted": n}                                # register/update
  POST /v1/bindings                                          # run-scoped binding
       body {"prefix_id": "...", "hint": {...partial AgentHint...}, "ttl_s": 600}
       -> 200 {"bound": 1}
       # the runner binds the current workflow run (session_id = run id) to a
       # node prefix; GET /v1/hints merges the active binding over the static
       # entry (binding wins). Bindings expire; they are not persisted.

  POST /agent-status                                         # lifecycle status sink
       body {"session": {"session_id": ...}, "agent_status": {...}}
       -> 200 {}
  GET  /healthz -> 200

Run: python -m hint_service --port 9200 --store hints.json
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path


class HintStore:
    """JSON-file backed store: prefix_id -> partial AgentHint. Thread-safe, atomic writes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._status: dict[str, dict] = {}
        # Run-scoped bindings: prefix -> {"hint": {...}, "expires_ms": int}.
        # Ephemeral (not persisted): a restart only loses one declaration.
        self._bindings: dict[str, dict] = {}
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

    def bind(self, prefix_id: str, hint: dict, ttl_s: int = 600) -> int:
        """Bind the current run's facts to a prefix (expires after ``ttl_s``)."""
        if not prefix_id or not isinstance(hint, dict):
            return 0
        try:
            ttl = int(ttl_s)
        except (TypeError, ValueError):
            ttl = 600
        ttl = max(1, ttl)
        expires_ms = int(time.time() * 1000) + ttl * 1000
        with self._lock:
            self._bindings[prefix_id] = {"hint": copy.deepcopy(hint), "expires_ms": expires_ms}
        return 1

    def query(self, prefix_id: str) -> dict:
        """Static hint merged with the active run-scoped binding (binding wins)."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            hint = copy.deepcopy(self._entries.get(prefix_id, {}))
            binding = self._bindings.get(prefix_id)
            if binding is not None:
                if binding.get("expires_ms", 0) <= now_ms:
                    self._bindings.pop(prefix_id, None)
                else:
                    _deep_merge(hint, binding.get("hint") or {})
            return hint

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
