"""HTTP layer: stdlib ThreadingHTTPServer; routes are documented in the package docstring."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .store import HintStore


def make_handler(store: HintStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "hint-service/0.1"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                self._send(200, {"ok": True})
                return
            if parsed.path == "/v1/hints":
                params = parse_qs(parsed.query)
                prefix_id = (params.get("prefix_id") or [""])[0].strip()
                self._send(200, store.query(prefix_id) if prefix_id else {})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid json"})
                return
            if not isinstance(body, dict):
                self._send(400, {"error": "body must be a json object"})
                return
            if parsed.path == "/v1/hints":
                entries = body.get("entries")
                if not isinstance(entries, list):
                    self._send(400, {"error": "entries must be a list"})
                    return
                self._send(200, {"upserted": store.upsert(entries)})
                return
            if parsed.path == "/agent-status":
                session = body.get("session") or {}
                agent_status = body.get("agent_status") or {}
                session_id = str((session or {}).get("session_id") or "").strip()
                if session_id and isinstance(agent_status, dict):
                    store.record_status(session_id, agent_status)
                self._send(200, {})
                return
            self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):  # noqa: A003 — keep default access logs off
            pass

    return Handler


def serve(host: str = "0.0.0.0", port: int = 9200, store: HintStore | None = None) -> ThreadingHTTPServer:
    store = store or HintStore()
    server = ThreadingHTTPServer((host, port), make_handler(store))
    return server
