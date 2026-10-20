# agent-hints (Dify plugin)

OpenAI-compatible model provider for [Dify](https://dify.ai/) that attaches
**agentic inference hints** to every model request, enabling hint-aware
inference gateways to do priority routing, KV-cache affinity and capacity
planning.

## The contract: X-Agent-Hint (full schema)

Every request carries an `X-Agent-Hint` HTTP header — the AgentHint object
serialized as UTF-8 JSON and encoded as base64url — with the **complete
agreed cross-harness schema**:

```jsonc
{
  "session": {
    "session_id": "…",                     // stable within one context
    "parent_session_id": "… | null",
    "derivation": { "relation": "fork|spawn|handoff", "join": "sync|async" },
    "agent_type": "…",                     // role / step identity
    "call_index": 1,                       // per-session call ordinal
    "epoch": 0                             // context version
  },
  "priority":        { "level": 1, "reason": "user_waiting" },
  "strict_priority": { "level": 1, "reason": "…" },   // control plane only
  "cache_control": [ { "type": "pin", "range": { "end": 1, "unit": "message" } } ],
  "prediction": {
    "latency_ms": 0, "remaining_calls": 0, "output_tokens": 0,
    "interval_ms": 0, "remaining_time_ms": 0, "expected_resume_s": 0,
    "next_action": { "state": "tool",
                     "tools": [{ "name": "web_search", "expected_duration_s": 8 }],
                     "prob": 0.85 }
  }
}
```

Field-by-field reference and data-source matrix:
**[docs/SCHEMA.md](docs/SCHEMA.md)**. Lifecycle states (`running / tool /
pause / suspend / finalize`) are reported on a separate status-event channel,
not inside model requests.

## The compatibility projection: nvext.agent_hints (NVIDIA Dynamo)

The same data is additionally projected into the request body as
`nvext.agent_hints`, compatible with
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) 1.0 agentic inference:

```json
{ "prefix_id": "ea80ab9a028ad311", "priority": 99,
  "osl": 256, "iat": 300, "latency_sensitivity": 2.5 }
```

`prefix_id` is a content hash of the prompt's static prefix (KV-cache
affinity); `priority` is converted from this plugin's "higher = more urgent"
scale to Dynamo's inverted one. Endpoints that ignore unknown fields are
unaffected.

## Data sources (low → high precedence)

1. **Auto defaults** — identity + `priority.level = 1`.
2. **Credential `hints_json`** — a static partial AgentHint JSON.
3. **Node completion params** — `priority / osl / iat / latency_sensitivity`
   tunable per LLM node in the Dify workflow UI.
4. **Hint service** (optional) — per-request dynamic enrichment; see below.

A fresh install is fully functional from tiers 1–3 (the scheduling basics +
`prefix_id`); the optional hint service enriches the remaining fields
(`call_index`, `next_action`, measured statistics, derivation) without any
plugin change. Outages degrade silently to a plain request.

## Quick start

1. Install the plugin (Dify → Plugins → GitHub: this repo, or Marketplace, or
   the `.difypkg` release asset).
2. Add a model from the **Agent Hints** provider: set the OpenAI-compatible
   `endpoint_url` (e.g. a Dynamo frontend) and optionally an `api_key`.
3. Bind the model in any workflow LLM node and tune the scheduling parameters
   in the node's parameter panel.

## Optional hint service

`GET {base}/v1/hints?prefix_id={hash}&user={user}&model={model}` → `200 OK`
with any partial AgentHint JSON (unknown fields ignored, highest merge
precedence). A **zero-dependency reference implementation** ships in this
repository under [`server/`](server/):

```bash
python -m hint_service --port 9200            # in-memory store
python -m hint_service --port 9200 --store hints.json  # JSON file store
```

Producers: your scheduling layer, CI pipelines, or compile-time tooling that
registers per-node metadata keyed by the same prefix hash the plugin derives.
Unreachable or malformed responses count as "no hints this time".

## Output switches & privacy

| Credential | Default | Effect |
|---|---|---|
| `emit_nvext` | true | Adds `nvext.agent_hints` to the request body. |
| `emit_header` | true | Adds the `X-Agent-Hint` header. |
| `cache_pin` | true | Pins the static system prefix (`cache_control`) unless hints provide one. |
| `cache_pin_ttl_s` | *(empty)* | Arms the auto pin with an explicit `ttl_s` (e.g. `600`) so TTL-bounded retention consumers — such as the Motor KV Hint bridge — can map the declaration; empty keeps the permanent pin. Invalid values degrade to the permanent pin. |

See [PRIVACY.md](PRIVACY.md). The plugin is stateless: it contacts only the
configured inference endpoint and (optionally) the configured hint service,
sending scheduling metadata (content hashes, integer estimates) — never
prompt content to the hint service.

## Development

```bash
python -m pytest tests/ -q     # 31 tests: schema golden (cross-language) + pipeline + service
```

Field naming and wire encoding are pinned by a cross-language golden vector;
see [docs/SCHEMA.md](docs/SCHEMA.md) for the schema contract.
