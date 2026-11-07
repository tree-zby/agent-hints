"""Dify integration (companion) tests: DSL mapping, 4-value status, bindings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from integrations.dify import (  # noqa: E402
    DifyRunBinder,
    lifecycle_status_for,
    load_dsl_info,
)
from integrations.dify.dsl import reachable_llm_nodes  # noqa: E402

DSL = """\
workflow:
  graph:
    nodes:
      - {id: start, data: {type: start}}
      - {id: a, data: {type: llm, step_id: step.a,
                       prompt_template: [{role: system, text: "alpha prefix"}]}}
      - {id: gate, data: {type: human-input, step_id: gate}}
      - {id: b, data: {type: llm, step_id: step.b,
                       prompt_template: [{role: system, text: "beta prefix"}]}}
      - {id: c, data: {type: llm, step_id: step.c,
                       prompt_template: [{role: system, text: "gamma prefix"}]}}
      - {id: end, data: {type: end}}
    edges:
      - {source: start, target: a, sourceHandle: source}
      - {source: a, target: gate, sourceHandle: source}
      - {source: gate, target: b, sourceHandle: approve}
      - {source: b, target: c, sourceHandle: source}
      - {source: c, target: end, sourceHandle: source}
"""


def _dsl_info(tmp_path: Path):
    path = tmp_path / "app.dsl.yml"
    path.write_text(DSL, encoding="utf-8")
    return load_dsl_info(path)


def test_dsl_mapping_and_reachable_llm_nodes(tmp_path: Path) -> None:
    prefixes, successors, node_types = _dsl_info(tmp_path)
    assert set(prefixes) == {"a", "b", "c"}
    assert node_types["gate"] == "human-input"
    assert successors["gate"] == ["b"]
    assert reachable_llm_nodes("gate", successors, node_types) == ["b", "c"]
    assert prefixes["b"] != prefixes["c"]


def test_status_mapping_folds_tool_and_suspend() -> None:
    started = lifecycle_status_for(
        {"event": "node_started", "data": {"node_type": "tool", "node_id": "t"}}
    )
    assert started is not None
    assert started["state"] == "pause"
    assert started["reason"] == "await_tool"
    assert started["tools"] == [{"name": "t"}]

    released = lifecycle_status_for(
        {"event": "node_finished", "data": {"node_type": "tool", "node_id": "t"}}
    )
    assert released is not None and released["state"] == "resume"

    pause = lifecycle_status_for(
        {"event": "node_started", "data": {"node_type": "human-input"}}
    )
    assert pause is not None and pause["state"] == "pause"
    resume = lifecycle_status_for(
        {"event": "node_finished", "data": {"node_type": "human-input"}}
    )
    assert resume is not None and resume["state"] == "resume"
    final = lifecycle_status_for(
        {"event": "workflow_finished", "data": {"status": "succeeded"}}
    )
    assert final is not None and final["state"] == "finalize"


def test_binder_prebinds_first_llm_resume_rest_running(tmp_path: Path) -> None:
    info = _dsl_info(tmp_path)
    binder = DifyRunBinder("http://hint-service", info)
    captured: list[tuple[str, str, str]] = []
    binder.bind = lambda prefix, run_id, state: captured.append((prefix, run_id, state)) or True  # type: ignore[method-assign]

    binder.on_human_input_required("gate", "run-1")
    prefixes, _, _ = info
    assert captured == [
        (prefixes["b"], "run-1", "resume"),
        (prefixes["c"], "run-1", "running"),
    ]


def test_binder_resume_is_one_shot(tmp_path: Path) -> None:
    info = _dsl_info(tmp_path)
    binder = DifyRunBinder("http://hint-service", info)
    captured: list[str] = []
    binder.bind = lambda prefix, run_id, state: captured.append(state) or True  # type: ignore[method-assign]

    binder.mark_resume()
    binder.on_node_started("a", "run-1")
    binder.on_node_started("a", "run-1")
    assert captured == ["resume", "running"]


def test_binder_posts_run_binding_payload(monkeypatch, tmp_path: Path) -> None:
    info = _dsl_info(tmp_path)
    binder = DifyRunBinder("http://hint-service", info)
    captured = []

    class _Response:
        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):  # noqa: ANN001, ARG001
        captured.append(request)
        return _Response()

    monkeypatch.setattr(
        "integrations.dify.bindings.urllib.request.urlopen", fake_urlopen
    )
    binder.bind(info[0]["a"], "run-42", "running")
    assert len(captured) == 1
    payload = json.loads(captured[0].data.decode("utf-8"))
    assert payload == {
        "prefix_id": info[0]["a"],
        "hint": {
            "session": {"session_id": "run-42"},
            "agent_status": {
                "state": "running",
                "state_since_ms": payload["hint"]["agent_status"]["state_since_ms"],
            },
        },
        "ttl_s": 600,
    }
