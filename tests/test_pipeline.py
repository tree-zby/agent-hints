"""Pipeline tests: four-tier merge, identity, nvext projection, degradation."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.conftest import hint_mod, llm_mod, pipeline_mod


def msg(role, content):
    return SimpleNamespace(role=role, content=content)


def sys_msg(content):
    return msg("system", content)


CREDS_BASE = {"endpoint_url": "http://ep/v1", "api_key": "k"}


# ---------------------------------------------------------------------------
# prefix_hash / identity
# ---------------------------------------------------------------------------
def test_prefix_hash_stable_and_distinct():
    a = [sys_msg("You are step A."), msg("user", "t")]
    b = [sys_msg("You are step A."), msg("user", "other")]
    c = [sys_msg("You are step B."), msg("user", "t")]
    assert pipeline_mod.prefix_hash(a) == pipeline_mod.prefix_hash(b)
    assert pipeline_mod.prefix_hash(a) != pipeline_mod.prefix_hash(c)
    assert len(pipeline_mod.prefix_hash(a)) == 16


def test_prefix_hash_fallback_and_empty():
    assert pipeline_mod.prefix_hash([msg("user", "hello")]) != "adhoc"
    assert pipeline_mod.prefix_hash([]) == "adhoc"


def test_identity_auto_prefers_user_then_prefix_hash():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE), user="run-1")
    assert hint.session.session_id == "run-1"
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE), user=None)
    assert hint.session.session_id == pipeline_mod.prefix_hash([sys_msg("s")])


def test_identity_user_mode_without_user_falls_back():
    creds = dict(CREDS_BASE, hint_identity="user")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, user=None)
    assert hint.session.session_id == pipeline_mod.prefix_hash([sys_msg("s")])


def test_identity_static_from_hints_json():
    creds = dict(CREDS_BASE, hint_identity="static",
                 hints_json=json.dumps({"session": {"session_id": "sess-9"}}))
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, user="ignored")
    assert hint.session.session_id == "sess-9"


# ---------------------------------------------------------------------------
# four-tier merge
# ---------------------------------------------------------------------------
def test_defaults_priority_level_1_and_no_extra_fields():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    assert hint.priority.level == 1
    assert hint.prediction is None


def test_credential_hints_json_merged():
    creds = dict(CREDS_BASE, hints_json=json.dumps(
        {"priority": {"level": 3, "reason": "background"},
         "prediction": {"output_tokens": 300}}))
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds)
    assert hint.priority.level == 3
    assert hint.priority.reason == "background"
    assert hint.prediction.output_tokens == 300


def test_invalid_hints_json_silently_ignored():
    creds = dict(CREDS_BASE, hints_json="not-json{{{")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds)
    assert hint.priority.level == 1


def test_node_params_override_credentials():
    creds = dict(CREDS_BASE, hints_json=json.dumps(
        {"priority": {"level": 3}, "prediction": {"output_tokens": 300}}))
    params = {"priority": 7, "osl": 128, "iat": 90}
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], params, creds)
    assert hint.priority.level == 7
    assert hint.prediction.output_tokens == 128
    assert hint.prediction.interval_ms == 90


def test_service_hints_override_node_params():
    def fake_fetch(url, timeout):
        assert "prefix_id=" in url and "user=run-2" in url and "model=m1" in url
        return {"session": {"agent_type": "research.web_search"},
                "prediction": {"output_tokens": 42,
                               "next_action": {"state": "tool",
                                               "tools": [{"name": "t"}], "prob": 1.0}}}

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    params = {"osl": 128}
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], params, creds,
                                      user="run-2", model="m1", fetch_fn=fake_fetch)
    assert hint.prediction.output_tokens == 42  # service > node param
    assert hint.prediction.next_action.state == "tool"
    assert hint.session.agent_type == "research.web_search"


def test_service_down_degrades_silently():
    def broken_fetch(url, timeout):
        raise RuntimeError("boom")

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, fetch_fn=broken_fetch)
    assert hint.priority.level == 1


def test_service_none_response_degrades():
    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds,
                                      fetch_fn=lambda url, t: None)
    assert hint.priority.level == 1


# ---------------------------------------------------------------------------
# cache_control
# ---------------------------------------------------------------------------
def test_cache_pin_default_and_off():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    assert hint.cache_control == [hint_mod.CacheControlHint(
        type="pin", range=hint_mod.HintRange(end=1, unit="message"))]
    creds = dict(CREDS_BASE, cache_pin="false")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds)
    assert hint.cache_control is None


def test_cache_control_from_service_not_overwritten():
    def fake_fetch(url, timeout):
        return {"cache_control": [{"type": "ephemeral", "ttl_s": 30}]}

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, fetch_fn=fake_fetch)
    assert hint.cache_control[0].type == "ephemeral"


# ---------------------------------------------------------------------------
# nvext projection
# ---------------------------------------------------------------------------
def test_nvext_defaults_and_prefix_id():
    hint, nvext = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    assert nvext == {"prefix_id": pipeline_mod.prefix_hash([sys_msg("s")]),
                     "priority": 99, "osl": 512, "iat": 250,
                     "latency_sensitivity": 1.0}


def test_nvext_priority_inverted():
    # our scale: higher = more urgent; Dynamo: lower = more priority
    hint, nvext = pipeline_mod.build_hint([sys_msg("s")], {"priority": 90},
                                          dict(CREDS_BASE))
    assert nvext["priority"] == 10
    hint, nvext = pipeline_mod.build_hint([sys_msg("s")], {"priority": 0},
                                          dict(CREDS_BASE))
    assert nvext["priority"] == 100


def test_nvext_reflects_service_prediction():
    def fake_fetch(url, timeout):
        return {"prediction": {"output_tokens": 42, "interval_ms": 17}}

    hint, nvext = pipeline_mod.build_hint([sys_msg("s")], {},
                                          dict(CREDS_BASE, hint_service_url="http://hs"),
                                          fetch_fn=fake_fetch)
    assert nvext["osl"] == 42
    assert nvext["iat"] == 17


# ---------------------------------------------------------------------------
# llm.py request assembly (headers/body)
# ---------------------------------------------------------------------------
def test_build_headers_emits_x_agent_hint():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    headers = llm_mod.build_headers(dict(CREDS_BASE), hint)
    assert headers["X-Agent-Hint"]
    assert headers["Authorization"] == "Bearer k"


def test_build_headers_emit_header_off():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {},
                                      dict(CREDS_BASE, emit_header="false"))
    headers = llm_mod.build_headers(dict(CREDS_BASE, emit_header="false"), hint)
    assert "X-Agent-Hint" not in headers


def test_build_body_nvext_toggle():
    messages = [{"role": "system", "content": "s"}]
    body = llm_mod.build_body("m", messages, {}, None, True, nvext={"prefix_id": "x"})
    assert body["nvext"] == {"agent_hints": {"prefix_id": "x"}}
    body = llm_mod.build_body("m", messages, {}, None, True, nvext={"prefix_id": "x"},
                              emit_nvext=False)
    assert "nvext" not in body


def test_build_body_hints_do_not_pollute_messages():
    messages = [{"role": "system", "content": "clean prompt"}]
    body = llm_mod.build_body("m", messages, {"temperature": 0.5}, ["stop1"], True,
                              nvext={"prefix_id": "x"})
    assert body["messages"] == messages  # prompt 不被改写
    assert body["stop"] == ["stop1"]
