# Agent Hint Schema — full field reference

The `X-Agent-Hint` header emitted by this plugin carries the **complete
agreed cross-harness schema**, field-for-field pinned by a cross-language
golden vector in `tests/test_schema_golden.py`.

Wire form: AgentHint object → UTF-8 JSON → base64url (unpadded) →
`X-Agent-Hint` HTTP header. The model request body stays a standard
OpenAI-compatible body (only the optional `nvext.agent_hints` scheduling
projection is added).

## Field tree

```jsonc
{
  "session": {
    "session_id": "…",                    // required
    "parent_session_id": "… | null",
    "derivation": { "relation": "fork|spawn|handoff",
                    "join": "sync|async",
                    "fork_point": 0 },
    "agent_type": "…",
    "call_index": 1,
    "epoch": 0
  },
  "agent_status": {                        // usually via the status channel, not the header
    "state": "running|tool|pause|suspend|finalize",
    "state_since_ms": 0,
    "reason": "await_tool|human_approval|…",
    "tools": [{ "name": "…", "call_id": "…", "child_session_id": "…" }]
  },
  "priority":        { "level": 1, "reason": "user_waiting|critical_path|background" },
  "strict_priority": { "level": 1, "reason": "…" },   // trusted control plane only
  "cache_control": [ { "type": "pin|ephemeral", "ttl_s": 0,
                       "range": { "start": 0, "end": 1, "unit": "message|token" } } ],
  "prediction": {
    "latency_ms": 0, "remaining_calls": 0, "output_tokens": 0,
    "interval_ms": 0, "remaining_time_ms": 0, "expected_resume_s": 0,
    "next_action": { "state": "running|tool|pause|suspend|finalize",
                     "tools": [{ "name": "…", "expected_duration_s": 0 }],
                     "prob": 0.0 }
  }
}
```

All fields are optional except `session.session_id`; absent sources mean
absent fields (`exclude_none`), never fabricated values.

## Where each field comes from

| Field | Plugin-local | Hint service | Notes |
|---|---|---|---|
| `session.session_id` | ✅ user channel / prefix hash / static | ✅ override | stable within one context |
| `session.agent_type` | ✅ `hints_json` | ✅ | role/step identity |
| `session.call_index` | — | ✅ | per-session call ordinal |
| `session.parent_session_id` / `derivation` | — | ✅ | multi-agent orchestration data |
| `session.epoch` | — | ✅ | context version after rewrites |
| `priority` | ✅ node param + `hints_json` | ✅ | higher = more urgent |
| `strict_priority` | — | ✅ (control plane) | agents never set it |
| `cache_control` | ✅ auto `pin(end:1,message)`, optional explicit `ttl_s` via `cache_pin_ttl_s` | ✅ | `cache_pin=false` disables the auto pin |
| `prediction.output_tokens` / `interval_ms` | ✅ node params (`osl` / `iat`) | ✅ | |
| `prediction.latency_ms` / `remaining_calls` / `remaining_time_ms` / `expected_resume_s` | — | ✅ | measured statistics |
| `prediction.next_action` | — | ✅ | compile-time/orchestrator knowledge |
| `agent_status` | — | — | **status event channel** (§4), not the request header |

## nvext projection (request body, Dynamo-compatible)

`nvext.agent_hints` is a **projection** of the same data for NVIDIA Dynamo:

| nvext field | source |
|---|---|
| `prefix_id` | content hash of the first system message (KV-cache affinity) |
| `priority` | `100 − priority.level` (Dynamo's scale is inverted: lower = more priority) |
| `osl` | `prediction.output_tokens` (default 512) |
| `iat` | `prediction.interval_ms` (default 250) |
| `latency_sensitivity` | node param / credential (default 1.0) |

Field names, types and encoding are contract-stable: any gateway built for
the agreed schema decodes this header unchanged.
