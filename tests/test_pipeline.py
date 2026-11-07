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
def test_defaults_two_level_baseline_and_no_extra_fields():
    # v2 tier-1 baseline: interactive posture (3 / user_waiting)
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    assert hint.priority.level == 3
    assert hint.priority.reason == "user_waiting"
    assert hint.prediction is None


def test_priority_baseline_from_trigger_source():
    """Plugin-native: a user turn is user-intervened; a continuation is not."""
    user_turn = [sys_msg("s"), msg("user", "question")]
    hint, _ = pipeline_mod.build_hint(user_turn, {}, dict(CREDS_BASE))
    assert hint.priority == hint_mod.PriorityHint(level=3, reason="user_waiting")

    continuation = [sys_msg("s"), msg("user", "q"), msg("tool", "result")]
    hint, _ = pipeline_mod.build_hint(continuation, {}, dict(CREDS_BASE))
    assert hint.priority == hint_mod.PriorityHint(level=1, reason="critical_path")


def test_priority_baseline_batch_run_mode():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE, run_mode="batch"))
    assert hint.priority == hint_mod.PriorityHint(level=1, reason="background")


def test_priority_reason_drives_level():
    continuation = [sys_msg("s"), msg("user", "q"), msg("tool", "r")]
    # An explicit level cannot contradict the reason: critical_path is level 1.
    hint, _ = pipeline_mod.build_hint(continuation, {"priority": 3}, dict(CREDS_BASE))
    assert hint.priority.level == 1
    assert hint.priority.reason == "critical_path"
    # user_waiting is the only reason that maps to level 3.
    creds = dict(CREDS_BASE, hints_json=json.dumps(
        {"priority": {"level": 3, "reason": "user_waiting"}}))
    hint, _ = pipeline_mod.build_hint(continuation, {}, creds)
    assert hint.priority.level == 3
    assert hint.priority.reason == "user_waiting"


def test_priority_level_folds_to_two_value_domain():
    cases = {1: 1, 2: 1, 3: 3, 4: 3, 0: 1, -5: 1, 7: 3}
    for level, expected in cases.items():
        assert pipeline_mod.normalize_priority_level(level) == expected


def test_fold_keeps_reason_from_lower_tier():
    creds = dict(CREDS_BASE, hints_json=json.dumps({"priority": {"reason": "background"}}))
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {"priority": 4}, creds)
    assert hint.priority.level == 1
    assert hint.priority.reason == "background"


def test_credential_hints_json_merged():
    creds = dict(CREDS_BASE, hints_json=json.dumps(
        {"priority": {"level": 3, "reason": "background"},
         "prediction": {"output_tokens": 300}}))
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds)
    assert hint.priority.level == 1
    assert hint.priority.reason == "background"
    assert hint.prediction.output_tokens == 300


def test_priority_reason_suffix_is_normalized():
    creds = dict(CREDS_BASE, hints_json=json.dumps(
        {"priority": {"level": 3, "reason": "background: context_compaction"}}))
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds)
    assert hint.priority.level == 1
    assert hint.priority.reason == "background"


def test_invalid_hints_json_silently_ignored():
    creds = dict(CREDS_BASE, hints_json="not-json{{{")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds)
    assert hint.priority.level == 3


def test_node_params_override_credentials():
    creds = dict(CREDS_BASE, hints_json=json.dumps(
        {"priority": {"level": 3}, "prediction": {"output_tokens": 300}}))
    params = {"priority": 7, "osl": 128, "iat": 90}
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], params, creds)
    assert hint.priority.level == 3  # node param 7 folds into the two-value domain
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
    assert hint.prediction.next_action.state == "pause"  # tool folds to pause
    assert hint.session.agent_type == "research.web_search"


def test_service_down_degrades_silently():
    def broken_fetch(url, timeout):
        raise RuntimeError("boom")

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, fetch_fn=broken_fetch)
    assert hint.priority.level == 3


def test_service_none_response_degrades():
    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds,
                                      fetch_fn=lambda url, t: None)
    assert hint.priority.level == 3


def test_state_fold_tool_suspend_and_unknown():
    assert pipeline_mod.fold_state("tool") == "pause"
    assert pipeline_mod.fold_state("suspend") == "pause"
    assert pipeline_mod.fold_state("running") == "running"
    assert pipeline_mod.fold_state("pause") == "pause"
    assert pipeline_mod.fold_state("resume") == "resume"
    assert pipeline_mod.fold_state("finalize") == "finalize"
    assert pipeline_mod.fold_state("weird") == "running"
    assert pipeline_mod.fold_state(None) == "running"


def test_service_agent_status_suspend_folds_to_pause():
    def fake_fetch(url, timeout):
        return {"agent_status": {"state": "suspend", "state_since_ms": 5,
                                 "reason": "checkpoint"}}

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, fetch_fn=fake_fetch)
    assert hint.agent_status.state == "pause"
    assert hint.agent_status.reason == "checkpoint"


# ---------------------------------------------------------------------------
# cache_control (CACHE_CONTROL_IMPLEMENTATION_GUIDE.md)
# ---------------------------------------------------------------------------
LONG = "cache block " * 120  # ~360 tokens


def test_cache_control_derived_default_and_off():
    hint, _ = pipeline_mod.build_hint([sys_msg(LONG)], {}, dict(CREDS_BASE))
    assert hint.cache_control == [hint_mod.CacheControlHint(
        type="ephemeral", ttl_s=600,
        range=hint_mod.HintRange(end=1, unit="message"))]
    creds = dict(CREDS_BASE, cache_pin="false")
    hint, _ = pipeline_mod.build_hint([sys_msg(LONG)], {}, creds)
    assert hint.cache_control is None


def test_cache_control_short_prefix_is_not_declared():
    hint, _ = pipeline_mod.build_hint([sys_msg("short")], {}, dict(CREDS_BASE))
    assert hint.cache_control is None


def test_cache_control_ttl_from_prediction_interval():
    creds = dict(CREDS_BASE,
                 hints_json=json.dumps({"prediction": {"interval_ms": 120000}}))
    hint, _ = pipeline_mod.build_hint([sys_msg(LONG)], {}, creds)
    assert hint.cache_control[0].type == "ephemeral"
    assert hint.cache_control[0].ttl_s == 240  # 120 s gap * 2


def test_cache_control_ttl_from_next_tool_expected_duration():
    creds = dict(CREDS_BASE, hints_json=json.dumps({
        "prediction": {"next_action": {
            "state": "tool",
            "tools": [{"name": "t", "expected_duration_s": 120}]}}}))
    hint, _ = pipeline_mod.build_hint([sys_msg(LONG)], {}, creds)
    assert hint.cache_control[0].ttl_s == 240


def test_cache_control_explicit_ttl_override_and_invalid_fallback():
    creds = dict(CREDS_BASE, cache_pin_ttl_s="600")
    hint, _ = pipeline_mod.build_hint([sys_msg(LONG)], {}, creds)
    assert hint.cache_control[0].ttl_s == 600
    for raw in ("abc", "0", "-5", ""):
        creds = dict(CREDS_BASE, cache_pin_ttl_s=raw)
        hint, _ = pipeline_mod.build_hint([sys_msg(LONG)], {}, creds)
        assert hint.cache_control[0].ttl_s == 600  # derived fallback


def test_cache_control_skips_prefix_rewrite():
    long_system = "rewrite block " * 120
    base = [sys_msg(long_system), msg("user", "one"), msg("user", "two")]
    hint, _ = pipeline_mod.build_hint(base, {}, dict(CREDS_BASE), user="sess-rewrite")
    assert hint.cache_control is not None
    shrunk = [sys_msg(long_system), msg("user", "one")]
    hint, _ = pipeline_mod.build_hint(shrunk, {}, dict(CREDS_BASE), user="sess-rewrite")
    assert hint.cache_control is None


def test_cache_control_distinct_prefixes_are_not_a_rewrite():
    """Two workflow nodes have different system prefixes; both must declare."""
    hint_a, _ = pipeline_mod.build_hint(
        [sys_msg("node A " + "x" * 600), msg("user", "one")],
        {}, dict(CREDS_BASE), user="run-multi",
    )
    hint_b, _ = pipeline_mod.build_hint(
        [sys_msg("node B " + "y" * 600), msg("user", "one")],
        {}, dict(CREDS_BASE), user="run-multi",
    )
    assert hint_a.cache_control is not None
    assert hint_b.cache_control is not None


def test_cache_control_from_service_not_overwritten():
    def fake_fetch(url, timeout):
        return {"cache_control": [{"type": "ephemeral", "ttl_s": 30}]}

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, fetch_fn=fake_fetch)
    assert hint.cache_control[0].type == "ephemeral"


# ---------------------------------------------------------------------------
# optional hint-service enrichment: lineage / fork / epoch / prediction
# ---------------------------------------------------------------------------
def test_service_lineage_spawn_enrichment():
    dynamic = {
        "session": {
            "parent_session_id": "s-root01",
            "derivation": {"relation": "spawn", "join": "sync"},
        }
    }
    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint(
        [sys_msg("s")], {}, creds, fetch_fn=lambda url, timeout: dynamic
    )
    assert hint.session.parent_session_id == "s-root01"
    assert hint.session.derivation == {"relation": "spawn", "join": "sync"}


def test_service_lineage_fork_and_epoch_enrichment():
    dynamic = {
        "session": {
            "parent_session_id": "s-root01",
            "derivation": {"relation": "fork", "fork_point": 12},
            "epoch": 3,
        }
    }
    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint(
        [sys_msg("s")], {}, creds, fetch_fn=lambda url, timeout: dynamic
    )
    assert hint.session.derivation == {"relation": "fork", "fork_point": 12}
    assert hint.session.epoch == 3


def test_service_prediction_latency_and_remaining_calls():
    dynamic = {"prediction": {"latency_ms": 4100, "remaining_calls": 3}}
    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint(
        [sys_msg("s")], {}, creds, fetch_fn=lambda url, timeout: dynamic
    )
    assert hint.prediction.latency_ms == 4100
    assert hint.prediction.remaining_calls == 3


# ---------------------------------------------------------------------------
# agent_status
# ---------------------------------------------------------------------------
def test_agent_status_defaults_to_running():
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    assert hint.agent_status is not None
    assert hint.agent_status.state == "running"
    assert isinstance(hint.agent_status.state_since_ms, int)


def test_agent_status_service_override_wins():
    def fake_fetch(url, timeout):
        return {"agent_status": {"state": "tool", "state_since_ms": 123,
                                 "reason": "await_tool",
                                 "tools": [{"name": "t", "call_id": "c"}]}}

    creds = dict(CREDS_BASE, hint_service_url="http://hs")
    hint, _ = pipeline_mod.build_hint([sys_msg("s")], {}, creds, fetch_fn=fake_fetch)
    assert hint.agent_status.state == "pause"  # tool wait folds to pause
    assert hint.agent_status.state_since_ms == 123
    assert hint.agent_status.reason == "await_tool"  # event detail kept


# ---------------------------------------------------------------------------
# nvext projection
# ---------------------------------------------------------------------------
def test_nvext_defaults_and_header_identity():
    hint, nvext = pipeline_mod.build_hint([sys_msg("s")], {}, dict(CREDS_BASE))
    assert nvext == {"priority": 3}
    assert hint.session.session_id == pipeline_mod.prefix_hash([sys_msg("s")])


@pytest.mark.parametrize("priority", [0, 1, 2, 3, 4, 5, 10, 90, "0", "5"])
def test_nvext_priority_folded_and_sent_unchanged(priority):
    # The wire level is reason-driven: user_waiting is 3, every continuation
    # is 1; nvext carries that folded level unchanged (no inversion).
    user_turn = [sys_msg("s"), msg("user", "q")]
    hint, nvext = pipeline_mod.build_hint(user_turn, {"priority": priority},
                                          dict(CREDS_BASE))
    assert nvext == {"priority": 3}
    assert type(nvext["priority"]) is int
    assert hint.priority.level == 3

    continuation = [sys_msg("s"), msg("user", "q"), msg("tool", "r")]
    hint, nvext = pipeline_mod.build_hint(continuation, {"priority": priority},
                                          dict(CREDS_BASE))
    assert nvext == {"priority": 1}
    assert hint.priority.level == 1


def test_service_prediction_stays_in_header():
    def fake_fetch(url, timeout):
        return {"prediction": {"output_tokens": 42, "interval_ms": 17}}

    hint, nvext = pipeline_mod.build_hint([sys_msg("s")], {},
                                          dict(CREDS_BASE, hint_service_url="http://hs"),
                                          fetch_fn=fake_fetch)
    assert nvext == {"priority": 3}
    headers = llm_mod.build_headers(dict(CREDS_BASE), hint)
    decoded = hint_mod.decode_agent_hint(headers["X-Agent-Hint"])
    assert decoded.prediction.output_tokens == 42
    assert decoded.prediction.interval_ms == 17


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
    prompt_messages = [sys_msg("s")]
    _, nvext = pipeline_mod.build_hint(prompt_messages, {}, dict(CREDS_BASE))
    messages = llm_mod.normalize_messages(prompt_messages)
    body = llm_mod.build_body("m", messages, {}, None, True, nvext=nvext)
    assert body["nvext"] == {"agent_hints": {"priority": 3}}
    body = llm_mod.build_body("m", messages, {}, None, True, nvext=nvext,
                              emit_nvext=False)
    assert "nvext" not in body


def test_build_body_hints_do_not_pollute_messages():
    messages = [{"role": "system", "content": "clean prompt"}]
    body = llm_mod.build_body("m", messages, {"temperature": 0.5}, ["stop1"], True,
                              nvext={"priority": 1})
    assert body["messages"] == messages  # prompt stays untouched
    assert body["stop"] == ["stop1"]
