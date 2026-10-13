# agent-hints (Dify plugin)

OpenAI-compatible model provider for [Dify](https://dify.ai/) that attaches
**agentic inference hints** to every model request, enabling hint-aware
inference gateways to do priority routing, KV-cache affinity and capacity
planning.

- **`nvext.agent_hints`** (request body) — compatible with
  [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) 1.0 agentic inference
  (`prefix_id` / `priority` / `osl` / `iat` / `latency_sensitivity`).
- **`X-Agent-Hint`** (HTTP header) — full-schema scheduling hints
  (UTF-8 JSON → base64url) for gateways that consume richer hints
  (session identity, prediction stats, next-action, cache control).

## Why

Inference orchestration layers (PD disaggregation, KV-aware routers) make
better scheduling decisions when requests declare their intent: expected
output length, arrival cadence, urgency. Agent frameworks like LangChain
already inject such hints (`ChatNVIDIADynamo`); this plugin brings the same
capability to Dify workflows — **per LLM node**, no code changes.

## Quick start

1. Install the plugin (Marketplace, GitHub repo, or `.difypkg`).
2. In your Dify workspace, add a model from the **Agent Hints** provider:
   set the OpenAI-compatible `endpoint_url` (e.g. a Dynamo frontend) and
   optionally an `api_key`.
3. Bind the model in any workflow LLM node. New tuning parameters appear in
   the node's parameter panel:

| Parameter | Default | Meaning |
|---|---|---|
| `priority` | 1 | Higher = more urgent. Converted to Dynamo's inverted priority scale automatically. |
| `osl` | 512 | Expected output tokens (capacity planning). |
| `iat` | 250 | Expected inter-arrival time in ms (load anticipation). |
| `latency_sensitivity` | 1.0 | Higher values get priority routing. |

With zero configuration the plugin still sends safe defaults and a
content-derived `prefix_id` (KV-cache affinity key: a stable hash of the
node's system prompt).

## Data sources (low → high precedence)

1. **Auto defaults** — identity + `priority.level = 1`.
2. **Credential `hints_json`** — a static partial AgentHint JSON, e.g.
   `{"priority": {"level": 3, "reason": "user_waiting"}, "prediction": {"output_tokens": 300}}`.
3. **Node completion params** — the four parameters above.
4. **Hint service** (optional) — set `hint_service_url` and the plugin issues
   `GET {url}/v1/hints?prefix_id=…&user=…&model=…` per request; the JSON
   response (any partial AgentHint) is merged with highest precedence.
   Unreachable/invalid responses degrade silently to a plain request.

Session identity (`session.session_id`): the `user` invocation parameter when
present, otherwise the prefix hash (`hint_identity` credential selects the
strategy).

## Output switches

| Credential | Default | Effect |
|---|---|---|
| `emit_nvext` | true | Adds `nvext.agent_hints` to the request body. |
| `emit_header` | true | Adds the `X-Agent-Hint` header. |
| `cache_pin` | true | Adds `cache_control: [{type: pin, range: {end: 1, unit: message}}]` (static system prefix) unless hints already provide `cache_control`. |

Requests to endpoints that ignore unknown fields are unaffected; everything
degrades to a standard OpenAI-compatible call.

## Hint service contract

`GET {base}/v1/hints?prefix_id={hash}&user={user}&model={model}` →
`200 OK` with any partial AgentHint JSON object (unknown fields ignored), e.g.:

```json
{
  "session": {"agent_type": "research.web_search", "call_index": 2},
  "prediction": {"output_tokens": 420, "interval_ms": 11800,
                 "next_action": {"state": "tool",
                                 "tools": [{"name": "web_search",
                                            "expected_duration_s": 8}],
                                 "prob": 0.85}}
}
```

Any non-200, timeout (default 2s, credential `hint_service_timeout`), or
malformed body is treated as "no hints this time".

## Privacy

See [PRIVACY.md](PRIVACY.md). The plugin sends user-provided prompt content
to the configured inference endpoint and, when configured, a `prefix_id`
(content hash), the invocation `user` value, and the model name to the
configured hint service. It stores nothing.

## Development

```bash
python -m pytest tests/ -q     # unit tests (schema golden + pipeline)
```

Schema authority: the golden vector in `tests/test_schema_golden.py` pins
field naming and wire encoding against the TS reference implementation
(AgentBundle `packages/core/src/agent-hint.ts`).
