# Agent Hint Schema — full field reference

The `X-Agent-Hint` header emitted by this plugin carries the **complete
agreed cross-harness schema**, field-for-field pinned by a cross-language
golden vector in `tests/test_schema_golden.py`.

Wire form: AgentHint object → UTF-8 JSON → base64url (unpadded) →
`X-Agent-Hint` HTTP header. The model request body stays a standard
OpenAI-compatible body (only the optional `nvext.agent_hints` scheduling
projection is added).

Auxiliary calls (compaction summaries, titles) carry no hint: pass the model
parameter `purpose=summary|compaction|title`; both the header and the optional
`nvext` projection are suppressed for that call.

## Field tree

```jsonc
{
  "session": {                             // required block
    "session_id": "…",                     // required
    "parent_session_id": "… | null",       // key always present; null on roots
    "derivation": { "relation": "fork|spawn|handoff",
                    "join": "sync|async",
                    "fork_point": 0 },
    "agent_type": "…",
    "call_index": 1,
    "epoch": 0,
    "state": "start|pause|resume|compact|append"   // required; `stop` is event-only
  },
  "context_management": {                   // required block
    "manage_request": "false",              // "true" = pure management request
    "edits": [{ "type": "offload|prefetch|evict", "target": "session" }]
  },
  "agent_status": {                         // 4-value wire contract
    "state": "running|pause|resume|finalize",
    "state_since_ms": 0,
    "reason": "await_tool|human_approval|…",
    "tools": [{ "name": "…", "call_id": "…", "child_session_id": "…" }]
  },
  "priority":         { "level": 1, "reason": "user_waiting|critical_path|background" },
  "priority_control": { "priority": { "level": 1, "reason": "…" } },  // required block
  "strict_priority":  { "level": 1, "reason": "…" },   // trusted control plane only
  "cache_control": {                        // required block, single object
    "type": "ephemeral", "ttl_s": 240, "msg_offset": 1,
    "range": { "end": 1, "unit": "message" }           // msg_offset == range.end
  },
  "prediction": {
    "latency_ms": 0, "remaining_calls": 0, "output_tokens": 0,
    "interval_ms": 0, "remaining_time_ms": 0, "expected_resume_s": 0,
    "next_action": { "state": "running|pause|resume|finalize",
                     "tools": [{ "name": "…", "expected_duration_s": 0 }],
                     "prob": 0.0 }
  }
}
```

`session`, `context_management`, `priority_control` and `cache_control` are the
required subset; a hint missing one of them is non-compliant. Producers may add
fields (they are ignored). `parent_session_id` is the one optional-looking key
that must exist: `null` says "this is a root", absence would be an unknown.
`cache_control` may be omitted only where the retention guide says "don't
declare"; `ttl_s` is advisory, never a promise.

`session.state` mapping used by this stateless plugin:

| value | when |
|---|---|
| `pause` | `agent_status.state == "pause"` (a tool wait or a human wait) |
| `compact` | the first request whose prefix was rewritten (count shrink / new first message) |
| `resume` | `agent_status.state == "resume"` (first request after a release) |
| `start` | a known first call (`call_index == 1`) of a root session |
| `append` | default |

`context_management.edits` is derived from that state — `pause → offload`,
`resume → prefetch`, `compact → evict`, otherwise empty — and `target` is
always `session`. Every request here is a request the agent was going to send,
so `manage_request` is always the string `"false"`.

Everything else is optional: absent sources mean absent fields
(`exclude_none`), never fabricated values.

## Where each field comes from

| Field | Plugin-local | Hint service | Notes |
|---|---|---|---|
| `session.session_id` | ✅ user channel / prefix hash / static (platform boundary: no run id in the plugin) | ✅ optional run-scoped binding | stands alone; the service only upgrades fidelity |
| `session.agent_type` | ✅ `hints_json` | ✅ | role/step identity |
| `session.call_index` | — | ✅ | per-session call ordinal |
| `session.parent_session_id` / `derivation` | — | ✅ | multi-agent orchestration data |
| `session.epoch` | — | ✅ | context version after rewrites |
| `session.state` | ✅ derived from `agent_status`, the prefix fingerprint and the call ordinal | — | six-value vocabulary; `stop` stays on the event channel |
| `context_management` | ✅ always present; edits expand from `session.state` | — | `manage_request` is always `"false"` here |
| `priority_control` | ✅ mirror of the normalized `priority` block | — | canonical location; top-level `priority` kept for compatibility |
| `priority` | ✅ plugin-native trigger classification + `run_mode` + node param + `hints_json` | ✅ optional step priority | reason-driven: `user_waiting` = 3, `critical_path`/`background` = 1; service/node values normalized to the same pair |
| `strict_priority` | — | ✅ (control plane) | agents never set it |
| `cache_control` | ✅ derived `ephemeral` + TTL per the guide; `cache_pin=false` disables; `cache_pin_ttl_s` overrides the derived TTL | ✅ (arrays and retired `pin` are normalized into one object) | single object: TTL = `clamp(gap*2, 60, 3600)`, `gap` from next-tool p90 / interval / 300 s; `msg_offset == range.end` = logical input count; omitted below 128 tokens or on a rewritten prefix |
| `prediction.output_tokens` / `interval_ms` | ✅ node params (`osl` / `iat`) | ✅ | |
| `prediction.latency_ms` / `remaining_calls` / `remaining_time_ms` / `expected_resume_s` | — | ✅ | measured statistics |
| `prediction.next_action` | — | ✅ | compile-time/orchestrator knowledge |
| `agent_status` | ✅ default `running` + `state_since_ms`; any producer's `tool`/`suspend` folds to `pause` | ✅ run-scoped override (incl. `resume`) | 4-value wire contract (`AGENT_STATUS_STATE_DESIGN.md` v1.0; `tool → pause` per the cross-harness adjudication); `reason`/`tools` ride the **status event channel** (§4) |

## Two-value level domain (v2)

`priority.level` is contracted to **{1, 3}**: `3` = user-intervened
(admission-exempt), `1` = everything else. The numbers are contract anchors —
the only semantics is the comparison against the inference-side admission
level (= 3), not tiers. The simplified contract is reason-driven: only
`user_waiting` is level 3; `critical_path` and `background` are level 1. Audit
reasons may carry a suffix in the wild (e.g.
`background: context_compaction`); the wire reason is the base word. A bare
level still folds into `{1,3}` so legacy callers keep their admission outcome.
`reason` remains the audit vocabulary
(`user_waiting` / `critical_path` / `background`).

## nvext projection (request body, Dynamo-compatible)

When explicitly enabled with `emit_nvext=true` (off by default so the model
request body stays hint-free), `nvext.agent_hints` contains only `priority`
for NVIDIA Dynamo v1.5.1:

| nvext field | source |
|---|---|
| `priority` | merged `priority.level`, folded into `{1, 3}` and sent unchanged as an integer (default `3` / `user_waiting`); higher values have higher priority |

There is no inversion or negation. The fold happens once, at the pipeline
exit, before both wire formats: legacy `{2, 4}` map to the nearest value
(`2 -> 1`, `4 -> 3`) and any other integer folds by the admission line
(`>= 3 -> 3`, else `-> 1`), so admission outcomes are unchanged.

Dynamo v1.5.1 accepts `priority`, `strict_priority`, `osl`,
`speculative_prefill` and `latency_sensitivity`, and rejects unknown fields.
This plugin sends only `priority`; `emit_nvext=false` omits `nvext` entirely.

`prefix_id` and `iat` are not Dynamo `AgentHints` fields. The prefix hash
remains in hint service queries and as the fallback `session.session_id` in
the header. `osl` and `iat` node parameters still populate the header's
`prediction.output_tokens` and `prediction.interval_ms`; neither is projected
into `nvext.agent_hints`. `latency_sensitivity` is no longer exposed or sent.

Version note: the former `100 - level` inversion matched Dynamo v1.0.x client
semantics. Since v1.1.0 the unified contract is "higher value = higher
priority" and clients must not negate it.

Field names, types and encoding are contract-stable: any gateway built for
the agreed schema decodes this header unchanged.
