"""hint_service tests: real-socket HTTP round trips + store semantics."""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from hint_service import HintStore, serve  # noqa: E402


@pytest.fixture()
def base_url(tmp_path):
    store = HintStore(tmp_path / "hints.json")
    server = serve("127.0.0.1", 0, store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", store
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_register_then_query_roundtrip(base_url):
    base, _ = base_url
    st, body = _post(base + "/v1/hints", {"entries": [
        {"prefix_id": "abc123", "hint": {
            "session": {"agent_type": "research.web_search", "call_index": 1},
            "prediction": {"next_action": {"state": "tool",
                                           "tools": [{"name": "web_search"}]},
                            "prob": 1.0}}}
    ]})
    assert (st, body) == (200, {"upserted": 1})
    st, hint = _get(base + "/v1/hints?prefix_id=abc123&user=run-1&model=m")
    assert st == 200
    assert hint["session"]["agent_type"] == "research.web_search"
    assert hint["prediction"]["next_action"]["tools"][0]["name"] == "web_search"


def test_query_unknown_prefix_returns_empty_object(base_url):
    base, _ = base_url
    st, hint = _get(base + "/v1/hints?prefix_id=unknown")
    assert (st, hint) == (200, {})


def test_upsert_merges_not_replaces(base_url):
    base, _ = base_url
    _post(base + "/v1/hints", {"entries": [
        {"prefix_id": "k", "hint": {"session": {"agent_type": "a"}, "priority": {"level": 2}}}]})
    _post(base + "/v1/hints", {"entries": [
        {"prefix_id": "k", "hint": {"session": {"call_index": 3}}}]})
    st, hint = _get(base + "/v1/hints?prefix_id=k")
    assert hint["session"] == {"agent_type": "a", "call_index": 3}
    assert hint["priority"] == {"level": 2}


def test_agent_status_endpoint_compat(base_url):
    base, store = base_url
    st, body = _post(base + "/agent-status", {
        "session": {"session_id": "run-1"},
        "agent_status": {"state": "finalize", "state_since_ms": 123}})
    assert (st, body) == (200, {})
    assert store.status_snapshot()["run-1"]["state"] == "finalize"


def test_store_persists_to_file(tmp_path):
    from hint_service import HintStore as S
    path = tmp_path / "h.json"
    s1 = S(path)
    s1.upsert([{"prefix_id": "p", "hint": {"session": {"agent_type": "x"}}}])
    s2 = S(path)
    assert s2.query("p")["session"]["agent_type"] == "x"


def test_healthz(base_url):
    base, _ = base_url
    st, body = _get(base + "/healthz")
    assert (st, body) == (200, {"ok": True})


def test_run_binding_overrides_static_entry(base_url):
    base, _ = base_url
    _post(base + "/v1/hints", {"entries": [
        {"prefix_id": "bind1", "hint": {
            "session": {"agent_type": "START", "call_index": 1},
            "priority": {"level": 3, "reason": "user_waiting"}}}]})
    st, body = _post(base + "/v1/bindings", {
        "prefix_id": "bind1",
        "hint": {"session": {"session_id": "run-42"},
                 "agent_status": {"state": "running", "state_since_ms": 1}},
        "ttl_s": 60})
    assert (st, body) == (200, {"bound": 1})
    st, hint = _get(base + "/v1/hints?prefix_id=bind1")
    assert hint["session"] == {
        "agent_type": "START", "call_index": 1, "session_id": "run-42"}
    assert hint["priority"]["reason"] == "user_waiting"
    assert hint["agent_status"]["state"] == "running"


def test_run_binding_expires(base_url, monkeypatch):
    base, _ = base_url
    import hint_service.store as store_mod

    now_ms = [1_000_000]
    monkeypatch.setattr(store_mod.time, "time", lambda: now_ms[0] / 1000.0)
    _post(base + "/v1/bindings", {
        "prefix_id": "exp",
        "hint": {"session": {"session_id": "r"}},
        "ttl_s": 10})
    st, hint = _get(base + "/v1/hints?prefix_id=exp")
    assert hint["session"]["session_id"] == "r"
    now_ms[0] += 11_000
    st, hint = _get(base + "/v1/hints?prefix_id=exp")
    assert hint == {}


def test_binding_requires_prefix_and_hint(base_url):
    base, _ = base_url
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base + "/v1/bindings", {"prefix_id": "", "hint": {}})
    assert exc.value.code == 400
