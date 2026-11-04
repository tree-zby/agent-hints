"""Run-scoped bindings for the Dify hint service.

Dify does not expose ``workflow_run_id`` to model plugins, so the workflow
driver binds the current run to the node's content prefix:

    POST {hint_service}/v1/bindings
    {"prefix_id": "...", "hint": {"session": {"session_id": run_id},
                                  "agent_status": {"state": "..."}}, "ttl_s": 600}

``GET /v1/hints?prefix_id=...`` merges the active binding over the static
registration. On approval, Dify no longer streams the resumed run to the
caller, so the driver pre-binds the reachable LLM nodes **before** submitting
the form: the first one carries ``resume`` (one-shot), the rest ``running``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .dsl import reachable_llm_nodes

DEFAULT_BINDING_TTL_S = 600
_STATES = ("running", "pause", "resume", "finalize")


class DifyRunBinder:
    """Bind the current workflow run to node prefixes (best-effort)."""

    def __init__(
        self,
        hint_service_url: str,
        dsl_info: tuple[dict[str, str], dict[str, list[str]], dict[str, str]] | None = None,
        *,
        ttl_s: int = DEFAULT_BINDING_TTL_S,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._url = hint_service_url.rstrip("/")
        self._prefixes, self._successors, self._node_types = dsl_info or ({}, {}, {})
        self._ttl_s = ttl_s
        self._resume_pending = False
        self._on_error = on_error

    def bind(self, prefix_id: str, run_id: str, state: str) -> bool:
        """POST one run-scoped binding; returns False on transport failure."""
        if not prefix_id or not run_id or state not in _STATES:
            return False
        payload = {
            "prefix_id": prefix_id,
            "hint": {
                "session": {"session_id": run_id},
                "agent_status": {
                    "state": state,
                    "state_since_ms": int(time.time() * 1000),
                },
            },
            "ttl_s": self._ttl_s,
        }
        try:
            request = urllib.request.Request(
                self._url + "/v1/bindings",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
            )
            request.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(request, timeout=2) as response:
                response.read()
            return True
        except (urllib.error.URLError, OSError) as exc:
            if self._on_error is not None:
                self._on_error(exc)
            return False

    def _take_state(self) -> str:
        state = "resume" if self._resume_pending else "running"
        self._resume_pending = False
        return state

    def on_node_started(self, node_id: str, run_id: str) -> None:
        """Bind a starting LLM node (resume is one-shot)."""
        prefix_id = self._prefixes.get(str(node_id or ""))
        if prefix_id:
            self.bind(prefix_id, run_id, self._take_state())

    def on_human_input_required(self, gate_node_id: str, run_id: str) -> None:
        """Pre-bind the post-approval LLM nodes (first resume, rest running).

        Must be called before submitting the approval form.
        """
        for index, node_id in enumerate(
            reachable_llm_nodes(str(gate_node_id or ""), self._successors, self._node_types)
        ):
            prefix_id = self._prefixes.get(node_id)
            if prefix_id:
                self.bind(prefix_id, run_id, "resume" if index == 0 else "running")

    def mark_resume(self) -> None:
        """Fallback: the next observed LLM node carries resume."""
        self._resume_pending = True

    @property
    def dsl_info(self) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
        return self._prefixes, self._successors, self._node_types
