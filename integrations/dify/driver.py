"""Minimal Dify workflow driver for hint field tests / evaluations.

Lives in the plugin companion (not the harness): it supplies the run-scoped
facts a Dify model plugin cannot see, without modifying the harness's runner.

- streams the workflow (``/v1/workflows/run``),
- binds the run to node prefixes (:class:`DifyRunBinder`),
- auto-passes human-input approvals via ``/v1/form/human_input/{token}``,
- reports 4-value ``agent_status`` events to a sink,
- pre-binds the post-approval nodes (Dify does not stream the resumed run).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .bindings import DifyRunBinder
from .dsl import load_dsl_info
from .status import lifecycle_status_for, post_status


class DifyWorkflowDriver:
    """Drive one Dify workflow run; best-effort bindings/status events."""

    def __init__(
        self,
        base: str,
        api_key: str,
        app_id: str,
        *,
        dsl_file: str | None = None,
        hint_service_url: str | None = None,
        status_endpoint: str | None = None,
        binding_ttl_s: int = 600,
    ) -> None:
        self.base = base.rstrip("/")
        self.api_key = api_key
        self.app_id = app_id
        self.status_endpoint = status_endpoint
        self.binder: DifyRunBinder | None = None
        if hint_service_url:
            dsl_info = load_dsl_info(dsl_file) if dsl_file else None
            self.binder = DifyRunBinder(hint_service_url, dsl_info, ttl_s=binding_ttl_s)

    # -- HTTP helpers --------------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: int = 60) -> None:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode("utf-8"), method="POST"
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()

    def _run_detail(self, run_id: str) -> dict[str, Any] | None:
        request = urllib.request.Request(self.base + "/v1/workflows/run/" + run_id)
        request.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode())
                return body.get("data") or body
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None

    def _report(
        self, run_id: str, status: dict[str, Any] | None, reported: set[str]
    ) -> None:
        if not status or not run_id or not self.status_endpoint:
            return
        post_status(self.status_endpoint, run_id, status)
        state = str(status.get("state") or "")
        if state:
            reported.add(state)

    # -- run -----------------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        user: str = "hint-e2e",
        auto_pass: bool = True,
        action: str = "approve",
        timeout: int = 300,
    ) -> dict[str, Any]:
        run_id = ""
        status = "running"
        outputs: Any = None
        form_token: str | None = None
        auto_submitted = False
        reported: set[str] = set()
        events: list[dict[str, Any]] = []

        payload = json.dumps(
            {"inputs": {"task": task}, "response_mode": "streaming", "user": user}
        ).encode()
        request = urllib.request.Request(
            self.base + "/v1/workflows/run", data=payload, method="POST"
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", "Bearer " + self.api_key)

        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                etype = event.get("event")
                data = event.get("data") or {}
                run_id = (
                    data.get("workflow_run_id") or event.get("workflow_run_id") or run_id
                )

                # Bind every reachable prefix to the run before the first model
                # request starts; node_started bindings only refine the state.
                if etype == "workflow_started" and self.binder is not None and run_id:
                    for prefix in self.binder.dsl_info[0].values():
                        self.binder.bind(prefix, run_id, "running")

                if etype == "node_started" and self.binder is not None and run_id:
                    self.binder.on_node_started(str(data.get("node_id") or ""), run_id)

                if etype == "human_input_required":
                    form_token = data.get("form_token")
                    if auto_pass and form_token:
                        # Pre-bind before submitting: the resumed run is not
                        # streamed back to this caller.
                        if self.binder is not None and run_id:
                            self.binder.on_human_input_required(
                                str(data.get("node_id") or ""), run_id
                            )
                        self._post_json(
                            "/v1/form/human_input/" + str(form_token),
                            {"action": action, "inputs": {}, "user": user},
                        )
                        auto_submitted = True

                agent_status = lifecycle_status_for(event)
                if agent_status is not None:
                    self._report(run_id, agent_status, reported)
                    events.append(agent_status)

                if etype == "workflow_finished":
                    status = "ok" if data.get("status") == "succeeded" else "error"
                    outputs = data.get("outputs")

        if auto_submitted and run_id and "resume" not in reported:
            resume = {"state": "resume", "state_since_ms": int(time.time() * 1000)}
            self._report(run_id, resume, reported)
            events.append(resume)

        if status == "running" and auto_submitted and run_id:
            for _ in range(60):
                time.sleep(2)
                detail = self._run_detail(run_id)
                if detail is None:
                    continue
                detail_status = detail.get("status")
                if detail_status == "succeeded":
                    status = "ok"
                    outputs = detail.get("outputs")
                    if "finalize" not in reported:
                        finalize = {
                            "state": "finalize",
                            "state_since_ms": int(time.time() * 1000),
                        }
                        self._report(run_id, finalize, reported)
                        events.append(finalize)
                    break
                if detail_status == "failed":
                    status = "error"
                    outputs = detail.get("outputs")
                    break

        return {"run_id": run_id, "status": status, "outputs": outputs, "events": events}


def run_workflow(
    base: str,
    api_key: str,
    app_id: str,
    task: str,
    *,
    dsl_file: str | None = None,
    hint_service_url: str | None = None,
    status_endpoint: str | None = None,
    user: str = "hint-e2e",
    auto_pass: bool = True,
    action: str = "approve",
    timeout: int = 300,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`DifyWorkflowDriver`."""
    driver = DifyWorkflowDriver(
        base,
        api_key,
        app_id,
        dsl_file=dsl_file,
        hint_service_url=hint_service_url,
        status_endpoint=status_endpoint,
    )
    return driver.run(
        task, user=user, auto_pass=auto_pass, action=action, timeout=timeout
    )
