"""Golden interop tests — cross-language pin for the Agent Hint schema.

Constants extracted verbatim from the ascend-llm plugin test suite (the
cross-language interop pin; same vector as the design doc §1.1 example).
"""
from tests.conftest import hint_mod  # noqa: F401

import base64
import json

TS_GOLDEN = (
    "eyJzZXNzaW9uIjp7InNlc3Npb25faWQiOiJzLTdmM2EyYyIsInBhcmVudF9zZXNzaW9uX2lkIjoi"
    "cy1yb290MDEiLCJkZXJpdmF0aW9uIjp7InJlbGF0aW9uIjoic3Bhd24iLCJqb2luIjoic3luYyJ9"
    "LCJhZ2VudF90eXBlIjoicmVzZWFyY2hlciIsImNhbGxfaW5kZXgiOjIsImVwb2NoIjowfSwiYWdl"
    "bnRfc3RhdHVzIjp7InN0YXRlIjoicnVubmluZyIsInN0YXRlX3NpbmNlX21zIjoxNzkwMDAwMDAw"
    "MDAwfSwicHJlZGljdGlvbiI6eyJyZW1haW5pbmdfY2FsbHMiOjMsIm91dHB1dF90b2tlbnMiOjQy"
    "MCwiaW50ZXJ2YWxfbXMiOjExODAwLCJsYXRlbmN5X21zIjo0MTAwLCJuZXh0X2FjdGlvbiI6eyJz"
    "dGF0ZSI6InRvb2wiLCJ0b29scyI6W3sibmFtZSI6IndlYl9zZWFyY2giLCJleHBlY3RlZF9kdXJh"
    "dGlvbl9zIjo4fV0sInByb2IiOjAuODV9fSwicHJpb3JpdHkiOnsibGV2ZWwiOjMsInJlYXNvbiI6"
    "ImNyaXRpY2FsX3BhdGgifSwiY2FjaGVfY29udHJvbCI6W3sidHlwZSI6InBpbiIsInJhbmdlIjp7"
    "ImVuZCI6MX19LHsidHlwZSI6ImVwaGVtZXJhbCIsInR0bF9zIjoxMiwicmFuZ2UiOnsiZW5kIjox"
    "MH19XX0"
)


TS_FULL_DICT = {
    "session": {
        "session_id": "s-7f3a2c",
        "parent_session_id": "s-root01",
        "derivation": {"relation": "spawn", "join": "sync"},
        "agent_type": "researcher",
        "call_index": 2,
        "epoch": 0,
    },
    "agent_status": {"state": "running", "state_since_ms": 1790000000000},
    "prediction": {
        "remaining_calls": 3,
        "output_tokens": 420,
        "interval_ms": 11800,
        "latency_ms": 4100,
        "next_action": {
            "state": "tool",
            "tools": [{"name": "web_search", "expected_duration_s": 8}],
            "prob": 0.85,
        },
    },
    "priority": {"level": 3, "reason": "critical_path"},
    "cache_control": [
        {"type": "pin", "range": {"end": 1}},
        {"type": "ephemeral", "ttl_s": 12, "range": {"end": 10}},
    ],
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


def test_header_name_matches_spec():
    assert hint_mod.AGENT_HINT_HEADER == "X-Agent-Hint"
