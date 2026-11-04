"""Dify SSE -> 4-value ``agent_status`` events (AGENT_STATUS_STATE_DESIGN v1.0).

``tool`` nodes fold to ``running`` (the reason/tools stay on the event) and
``suspend`` folds to ``pause``; a human-input completion is the one-shot
``resume`` edge.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

_TOOL_FAMILIES = {"tool", "code", "http-request"}
_WIRE_STATES = {"running", "pause", "resume", "finalize"}


def lifecycle_status_for(evt: dict[str, Any]) -> dict[str, Any] | None:
    """Map one Dify SSE event to a 4-value ``agent_status`` object, or None."""
    etype = evt.get("event")
    data = evt.get("data") or {}
    now_ms = int(time.time() * 1000)

    if etype == "workflow_started":
        return {"state": "running", "state_since_ms": now_ms}

    if etype in ("node_started", "node_finished"):
        node_type = str(data.get("node_type") or "")
        started = etype == "node_started"
        if node_type == "human-input":
            if started:
                return {
                    "state": "pause",
                    "state_since_ms": now_ms,
                    "reason": "human_approval",
                }
            return {"state": "resume", "state_since_ms": now_ms}
        if node_type in _TOOL_FAMILIES:
            if started:
                name = str(
                    data.get("step_id") or data.get("node_id") or data.get("title") or node_type
                )
                tool_ref: dict[str, Any] = {"name": name}
                call_id = data.get("execution_id") or data.get("id")
                if call_id:
                    tool_ref["call_id"] = str(call_id)
                # tool folds to running on the wire; the ref stays on the event.
                return {
                    "state": "running",
                    "state_since_ms": now_ms,
                    "reason": "await_tool",
                    "tools": [tool_ref],
                }
            return {"state": "running", "state_since_ms": now_ms}
        return None

    if etype == "workflow_finished":
        return {
            "state": "finalize",
            "state_since_ms": now_ms,
            "reason": "failed" if data.get("status") == "failed" else None,
        }
    return None


def post_status(endpoint: str, run_id: str, agent_status: dict[str, Any]) -> bool:
    """POST ``{"session": ..., "agent_status": ...}`` to the status sink."""
    if not endpoint or not run_id:
        return False
    payload = {"session": {"session_id": run_id}, "agent_status": agent_status}
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        return True
    except (urllib.error.URLError, OSError):
        return False
