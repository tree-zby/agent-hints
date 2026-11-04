# Dify integration (companion, not packaged into the plugin)

Dify's model plugin API exposes only the request (messages, parameters,
credentials, user). It cannot see the workflow run id, node lifecycle or human
approvals. This companion supplies those facts **without touching the harness**:

| Module | Purpose |
|---|---|
| `dsl.py` | app DSL -> `node_id -> prefix_id` (content hash of the first system template), graph successors/types |
| `bindings.py` | `POST {hint_service}/v1/bindings` for the current run; approval pre-binding (first post-gate LLM = `resume`, rest = `running`) |
| `status.py` | Dify SSE -> 4-value `agent_status` (`tool` -> `running`, `suspend` -> `pause`, human-input completion -> `resume`); posts to a status sink |
| `driver.py` | minimal workflow driver (SSE, auto-pass form submit, bindings, status events) for tests/evaluations |

## Why a driver?

Dify does not stream the **resumed** run back to the caller after an approval,
so a driver must pre-bind the post-approval nodes before submitting the form.
This driver is the reference producer; any evaluation harness can reuse it or
implement the same calls.

## Usage

```python
from integrations.dify import DifyWorkflowDriver

driver = DifyWorkflowDriver(
    base="http://localhost",
    api_key=api_key,
    app_id=app_id,
    dsl_file="tests/fixtures/hint-approval-e2e.dsl.yml",
    hint_service_url="http://localhost:9200",
    status_endpoint="http://localhost:9999/agent-status",
)
result = driver.run("整理纪要", user="e2e", auto_pass=True, action="approve")
```

Requirements: Python stdlib + **PyYAML** (`pip install pyyaml`) for `dsl.py`.

## Boundary

- The plugin package (`models/`, `provider/`, `manifest.yaml`) is standalone and
  works without this companion (session falls back to the platform user, state
  is `running`, priority/cache are plugin-native).
- This companion is deployment/test infrastructure: it is **excluded** from the
  `.difypkg` and is not installed into Dify.
- System templates must stay static for the content-hash join to resolve.
