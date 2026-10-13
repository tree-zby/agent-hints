# Privacy Policy

The agent-hints plugin processes data solely to attach scheduling hints to
model requests made on behalf of the Dify workspace that installed it.

## What the plugin sends

1. **To the configured inference endpoint** (`endpoint_url`): the standard
   OpenAI-compatible chat request — user-provided prompt content and model
   parameters — plus scheduling hint metadata it derives itself:
   - `nvext.agent_hints` (request body): a content hash of the first system
     message (`prefix_id`), integer/float scheduling estimates
     (`priority`, `osl`, `iat`, `latency_sensitivity`).
   - `X-Agent-Hint` (header, optional): the same estimates plus session
     identity (content hash or the invocation `user` value), optional cache
     control markers, and any hint data returned by the configured hint
     service.

2. **To the configured hint service** (`hint_service_url`, optional): three
   query parameters — `prefix_id` (content hash, not raw content), `user`
   (the invocation `user` string passed by Dify), and `model` (the model
   name). The plugin never sends prompt content to the hint service.

## What the plugin stores

Nothing. The plugin is stateless: no data is written to disk, no logs of
request content are kept. Credential values are stored by Dify's own plugin
credential store.

## Third parties

The plugin contacts exactly two destinations chosen by the workspace
administrator: the configured inference endpoint and the configured hint
service. No telemetry, no other network calls.
