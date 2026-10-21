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

The request body includes only `priority` in `nvext.agent_hints`, matching
the [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) v1.5.1 request contract:

```json
{ "nvext": { "agent_hints": { "priority": 1 } } }
```

`priority.level` is sent unchanged as an integer: higher values have higher
priority. There is no inversion or negation; 0 stays 0, and the default is 1.
The four-tier merge below still determines the final priority.

Dynamo v1.5.1 rejects unknown `AgentHints` fields. This plugin sends no
`prefix_id`, `iat`, `osl` or `latency_sensitivity` in `nvext.agent_hints`,
even when prediction metadata is configured. The prefix hash still identifies
hint service queries and supplies the fallback `session.session_id` in the
`X-Agent-Hint` header. Prediction metadata remains available in that header.

## Data sources (low → high precedence)

1. **Auto defaults** — identity + `priority.level = 1`.
2. **Credential `hints_json`** — a static partial AgentHint JSON.
3. **Node completion params** — `priority / osl / iat` tunable per LLM node
   in the Dify workflow UI. `osl` and `iat` affect only the header.
4. **Hint service** (optional) — per-request dynamic enrichment; see below.

A fresh install is fully functional from tiers 1–3 (priority and header
metadata); the optional hint service enriches the remaining fields
(`call_index`, `next_action`, measured statistics, derivation) without any
plugin change. Outages degrade silently to a plain request.

## Quick start

1. Install the plugin (Dify → Plugins → GitHub: this repo, or Marketplace, or
   the `.difypkg` release asset).
2. Add a model from the **Agent Hints** provider: set the OpenAI-compatible
   `endpoint_url` (e.g. a Dynamo frontend) and optionally an `api_key`.
3. Bind the model in any workflow LLM node and tune priority and generation
   parameters in the node's parameter panel.

`temperature`, `max_tokens` and `top_p` are optional generation parameters.
Supplied values go into the OpenAI-compatible request body; absent or `null`
values are omitted. The plugin does not insert UI defaults for absent parameters.

| Parameter | Range | UI default |
|---|---|---|
| `temperature` | 0–2 | 1.0 |
| `max_tokens` | 1–configured context size | 512, capped by context size |
| `top_p` | 0–1 | 1.0 |

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
| `emit_nvext` | true | Adds `nvext.agent_hints` with only `priority`; false omits `nvext`. |
| `emit_header` | true | Adds the `X-Agent-Hint` header. |
| `cache_pin` | true | Pins the static system prefix (`cache_control`) unless hints provide one. |

See [PRIVACY.md](PRIVACY.md). The plugin is stateless: it contacts only the
configured inference endpoint and (optionally) the configured hint service,
sending scheduling metadata (content hashes, integer estimates) — never
prompt content to the hint service.

## Development

```bash
python -m pytest -q -p no:cacheprovider
```

Tests cover the schema golden vector, pipeline, hint service and serialized
requests captured through the Dify model invocation path with a mock HTTP
transport. To print the priority table and a full captured request body:

```bash
python -m tests.test_request_contract
```

Field naming and wire encoding are pinned by a cross-language golden vector;
see [docs/SCHEMA.md](docs/SCHEMA.md) for the schema contract.
