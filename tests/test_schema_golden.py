"""Golden interop tests — cross-language pin for the Agent Hint schema.

The vector was re-pinned for the v1 ordinary-request profile: an ordinary hint
carries `session` (state `start`/`append`) + `priority_control` + an optional
single-object `cache_control`; `context_management` belongs to manage requests
only and is absent here. Encoding/decoding behavior is unchanged (UTF-8 JSON
-> base64url).
"""
from tests.conftest import hint_mod  # noqa: F401

import base64
import json

TS_GOLDEN = (
    "eyJzZXNzaW9uIjp7InNlc3Npb25faWQiOiJzLTdmM2EyYyIsInBhcmVudF9zZXNzaW9uX2lkIjoi"
    "cy1yb290MDEiLCJkZXJpdmF0aW9uIjp7InJlbGF0aW9uIjoic3Bhd24iLCJqb2luIjoic3luYyJ9"
    "LCJhZ2VudF90eXBlIjoicmVzZWFyY2hlciIsImNhbGxfaW5kZXgiOjIsImVwb2NoIjowLCJzdGF0"
    "ZSI6ImFwcGVuZCJ9LCJhZ2VudF9zdGF0dXMiOnsic3RhdGUiOiJydW5uaW5nIiwic3RhdGVfc2lu"
    "Y2VfbXMiOjE3OTAwMDAwMDAwMDB9LCJwcmlvcml0eSI6eyJsZXZlbCI6MywicmVhc29uIjoiY3Jp"
    "dGljYWxfcGF0aCJ9LCJwcmlvcml0eV9jb250cm9sIjp7InByaW9yaXR5Ijp7ImxldmVsIjozLCJy"
    "ZWFzb24iOiJjcml0aWNhbF9wYXRoIn19LCJjYWNoZV9jb250cm9sIjp7InR5cGUiOiJlcGhlbWVy"
    "YWwiLCJ0dGxfcyI6MTIsIm1zZ19vZmZzZXQiOjEwLCJyYW5nZSI6eyJlbmQiOjEwLCJ1bml0Ijoi"
    "bWVzc2FnZSJ9fSwicHJlZGljdGlvbiI6eyJsYXRlbmN5X21zIjo0MTAwLCJyZW1haW5pbmdfY2Fs"
    "bHMiOjMsIm91dHB1dF90b2tlbnMiOjQyMCwiaW50ZXJ2YWxfbXMiOjExODAwLCJuZXh0X2FjdGlv"
    "biI6eyJzdGF0ZSI6InJ1bm5pbmciLCJ0b29scyI6W3sibmFtZSI6IndlYl9zZWFyY2giLCJleHBl"
    "Y3RlZF9kdXJhdGlvbl9zIjo4LjB9XSwicHJvYiI6MC44NX19fQ"
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
    )
    encoded = hint_mod.encode_agent_hint(hint)
    padded = encoded + "=" * (-len(encoded) % 4)
    wire = json.loads(base64.urlsafe_b64decode(padded))
    assert "parent_session_id" in wire["session"]
    assert wire["session"]["parent_session_id"] is None
    assert wire["session"]["state"] == "start"
    assert "context_management" not in wire


def test_missing_required_blocks_are_rejected():
    """Missing = non-compliant, not "default to something reasonable"."""
    try:
        hint_mod.AgentHint.model_validate({})
    except Exception as exc:  # pydantic.ValidationError
        assert "session" in str(exc)
    else:  # pragma: no cover - the assertion below explains the failure
        raise AssertionError("a hint without the required blocks must not validate")

    try:
        hint_mod.AgentHint.model_validate({"session": {"session_id": "s"}})
    except Exception as exc:  # pydantic.ValidationError
        assert "session.state" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("session.state is mandatory")


def test_manage_hint_carries_boolean_flag_and_one_explicit_edit():
    """Manage requests: `manage_request: true` + exactly one typed edit."""
    hint = hint_mod.build_manage_hint(
        session_id="s-7f3a2c",
        state="compact",
        call_index=2,
        epoch=0,
    )
    encoded = hint_mod.encode_agent_hint(hint)
    padded = encoded + "=" * (-len(encoded) % 4)
    wire = json.loads(base64.urlsafe_b64decode(padded))
    assert wire["session"]["state"] == "compact"
    assert wire["context_management"] == {
        "manage_request": True,
        "edits": [{"type": "evict", "target": "session"}],
    }
    assert "priority_control" not in wire
    assert "cache_control" not in wire
    assert "agent_status" not in wire


def test_manage_request_must_be_boolean_true_and_exactly_one_edit():
    for value in ("true", "false", False):
        try:
            hint_mod.AgentHint.model_validate(
                {
                    "session": {"session_id": "s", "state": "pause"},
                    "context_management": {
                        "manage_request": value,
                        "edits": [{"type": "offload"}],
                    },
                }
            )
        except Exception:  # pydantic.ValidationError
            continue
        raise AssertionError(f"manage_request={value!r} must not validate")

    try:
        hint_mod.AgentHint.model_validate(
            {
                "session": {"session_id": "s", "state": "pause"},
                "context_management": {"manage_request": True, "edits": []},
            }
        )
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("a manage hint must carry exactly one edit")


def test_header_name_matches_spec():
    assert hint_mod.AGENT_HINT_HEADER == "X-Agent-Hint"
