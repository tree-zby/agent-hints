"""Dify integration for the agent-hints plugin (companion package).

This directory is **not** part of the Dify plugin package: it is the
deployment/test-side integration used by a workflow driver to supply the
run-scoped facts the model plugin cannot see (run id, approval resume).

- :func:`load_dsl_info` — Dify app DSL -> node prefix / graph shape
- :class:`DifyRunBinder` — POST /v1/bindings for the current run
- :func:`lifecycle_status_for` — Dify SSE -> 4-value ``agent_status``
- :class:`DifyWorkflowDriver` — minimal SSE driver (auto-pass approvals)
"""

from .bindings import DifyRunBinder
from .driver import DifyWorkflowDriver, run_workflow
from .dsl import load_dsl_info
from .status import lifecycle_status_for, post_status

__all__ = [
    "DifyRunBinder",
    "DifyWorkflowDriver",
    "lifecycle_status_for",
    "load_dsl_info",
    "post_status",
    "run_workflow",
]
