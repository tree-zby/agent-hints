"""Golden interop tests — cross-language pin for the Agent Hint schema.

The vector was re-pinned when the four-block contract landed: `session.state`,
`context_management`, `priority_control` and a single-object `cache_control`
are now required, so the previous array/`pin` vector no longer describes a
valid hint. Encoding/decoding behavior is unchanged (UTF-8 JSON -> base64url).
"""
from tests.conftest import hint_mod  # noqa: F401

import base64
import json

TS_GOLDEN = (
    "eyJzZXNzaW9uIjp7InNlc3Npb25faWQiOiJzLTdmM2EyYyIsInBhcmVudF9zZXNzaW9uX2lkIjoi"
    "cy1yb290MDEiLCJkZXJpdmF0aW9uIjp7InJlbGF0aW9uIjoic3Bhd24iLCJqb2luIjoic3luYyJ9"
    "LCJhZ2VudF90eXBlIjoicmVzZWFyY2hlciIsImNhbGxfaW5kZXgiOjIsImVwb2NoIjowLCJzdGF0"
    "ZSI6ImFwcGVuZCJ9LCJjb250ZXh0X21hbmFnZW1lbnQiOnsibWFuYWdlX3JlcXVlc3QiOiJmYWxz"
    "ZSIsImVkaXRzIjpbXX0sImFnZW50X3N0YXR1cyI6eyJzdGF0ZSI6InJ1bm5pbmciLCJzdGF0ZV9z"
    "aW5jZV9tcyI6MTc5MDAwMDAwMDAwMH0sInByZWRpY3Rpb24iOnsicmVtYWluaW5nX2NhbGxzIjoz"
    "LCJvdXRwdXRfdG9rZW5zIjo0MjAsImludGVydmFsX21zIjoxMTgwMCwibGF0ZW5jeV9tcyI6NDEw"
    "MCwibmV4dF9hY3Rpb24iOnsic3RhdGUiOiJydW5uaW5nIiwidG9vbHMiOlt7Im5hbWUiOiJ3ZWJf"
    "c2VhcmNoIiwiZXhwZWN0ZWRfZHVyYXRpb25fcyI6OH1dLCJwcm9iIjowLjg1fX0sInByaW9yaXR5"
    "Ijp7ImxldmVsIjozLCJyZWFzb24iOiJjcml0aWNhbF9wYXRoIn0sInByaW9yaXR5X2NvbnRyb2wi"
    "OnsicHJpb3JpdHkiOnsibGV2ZWwiOjMsInJlYXNvbiI6ImNyaXRpY2FsX3BhdGgifX0sImNhY2hl"
    "X2NvbnRyb2wiOnsidHlwZSI6ImVwaGVtZXJhbCIsInR0bF9zIjoxMiwibXNnX29mZnNldCI6MTAs"
    "InJhbmdlIjp7ImVuZCI6MTAsInVuaXQiOiJtZXNzYWdlIn19fQ"
)


TS_FULL_DICT = {
    "session": {
        "session_id": "s-7f3a2c",
        "parent_session_id": "s-root01",
        "derivation": {"relation": "spawn", "join": "sync"},
        "agent_type": "researcher",
        "call_index": 2,
        "epoch": 0,
        "state": "append",
    },
    "context_management": {"manage_request": "false", "edits": []},
    "agent_status": {"state": "running", "state_since_ms": 1790000000000},
    "prediction": {
        "remaining_calls": 3,
        "output_tokens": 420,
        "interval_ms": 11800,
        "latency_ms": 4100,
        "next_action": {
            "state": "running",
            "tools": [{"name": "web_search", "expected_duration_s": 8}],
            "prob": 0.85,
        },
    },
    "priority": {"level": 3, "reason": "critical_path"},
    "priority_control": {"priority": {"level": 3, "reason": "critical_path"}},
    "cache_control": {
        "type": "ephemeral",
        "ttl_s": 12,
        "msg_offset": 10,
        "range": {"end": 10, "unit": "message"},
    },
}


def test_golden_vector_from_ts_decodes_identically():
    hint = hint_mod.decode_agent_hint(TS_GOLDEN)
    assert hint.model_dump(exclude_none=True) == TS_FULL_DICT


def test_python_encode_roundtrip_matches_ts_semantics():
    hint = hint_mod.AgentHint.model_validate(TS_FULL_DICT)
    decoded = hint_mod.decode_agent_hint(hint_mod.encode_agent_hint(hint))
    assert decoded.model_dump(exclude_none=True) == TS_FULL_DICT


def test_wire_form_is_base64url_unpadded_utf8_json():
    encoded = hint_mod.encode_agent_hint(hint_mod.AgentHint.model_validate(TS_FULL_DICT))
    assert "=" not in encoded
    assert "+" not in encoded and "/" not in encoded
    padded = encoded + "=" * (-len(encoded) % 4)
    assert json.loads(base64.urlsafe_b64decode(padded))["session"]["session_id"] == "s-7f3a2c"


def test_root_session_publishes_explicit_null_parent():
    """`exclude_none` must not drop the key: null means "this is a root"."""
    hint = hint_mod.AgentHint(
        session={"session_id": "s-root", "state": "start"},
        context_management={"manage_request": "false", "edits": []},
    )
    encoded = hint_mod.encode_agent_hint(hint)
    padded = encoded + "=" * (-len(encoded) % 4)
    wire = json.loads(base64.urlsafe_b64decode(padded))
    assert "parent_session_id" in wire["session"]
    assert wire["session"]["parent_session_id"] is None
    assert wire["session"]["state"] == "start"
    assert wire["context_management"] == {"manage_request": "false", "edits": []}


def test_missing_required_blocks_are_rejected():
    """Missing = non-compliant, not "default to something reasonable"."""
    try:
        hint_mod.AgentHint.model_validate({"session": {"session_id": "s"}})
    except Exception as exc:  # pydantic.ValidationError
        message = str(exc)
        assert "session.state" in message
        assert "context_management" in message
    else:  # pragma: no cover - the assertion below explains the failure
        raise AssertionError("a hint without the required blocks must not validate")


def test_header_name_matches_spec():
    assert hint_mod.AGENT_HINT_HEADER == "X-Agent-Hint"
