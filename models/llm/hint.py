"""Agent Hint schema — field-for-field mirror of the agreed cross-harness
specification.

Field naming and wire encoding are pinned by a cross-language golden vector
in tests/test_schema_golden.py. Wire form: the AgentHint object is serialized
as UTF-8 JSON and encoded as base64url into the `X-Agent-Hint` HTTP header;
the model request body keeps its original format (nvext.agent_hints is a
separate, Dynamo-compatible projection produced by the pipeline, not part of
this schema).

The v1 protocol splits requests into three kinds (protocol §1.2/§5):

  ordinary request    session + priority_control (+ cache_control when qualified)
  auxiliary request   no hint at all (suppressed by `purpose=`)
  manage request      session + context_management (lifecycle verb)

This model provider only ever prepares requests the agent was already going
to send, so it emits **ordinary** hints: `context_management` is deliberately
absent, and `session.state` is restricted to `start` / `append`. Lifecycle
verbs (`pause` / `resume` / `stop` / `compact`) belong on separate manage
requests, which need a lifecycle trigger this plugin does not have (L1
degradation, protocol §12.4). The `ContextManagement` model is still defined
so manage hints decode/validate correctly.

`priority` (top-level) stays as a compatibility mirror of
`priority_control.priority`, and `agent_status` keeps carrying the 4-value
lifecycle word used by the event channel (a tolerated superset field).

See docs/SCHEMA.md for the full field reference.
"""
import base64
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

AGENT_HINT_HEADER = "X-Agent-Hint"

# Per-request lifecycle summary. `stop` is reserved for the lifecycle channel
# and is never sent on a model request by this plugin.
SessionState = Literal["start", "pause", "resume", "stop", "compact", "append"]


class HintToolRef(BaseModel):
    name: str
    call_id: Optional[str] = None
    child_session_id: Optional[str] = None


class HintPendingTool(BaseModel):
    name: str
    expected_duration_s: Optional[float] = None


class SessionHint(BaseModel):
    session_id: str
    # Explicit null on root sessions: the key must exist on every request.
    parent_session_id: Optional[str] = None
    derivation: Optional[dict] = None
    agent_type: Optional[str] = None
    call_index: Optional[int] = None
    epoch: Optional[int] = None
    state: SessionState

    @field_validator("session_id")
    @classmethod
    def _session_id_required(cls, value: str) -> str:
        if not value:
            raise ValueError("session.session_id must be non-empty")
        return value


class AgentStatusHint(BaseModel):
    state: Literal["running", "pause", "resume", "finalize"]
    state_since_ms: Optional[int] = None
    reason: Optional[str] = None
    tools: Optional[list[HintToolRef]] = None


class PriorityHint(BaseModel):
    level: int
    reason: Optional[str] = None


class PriorityControl(BaseModel):
    """Canonical location of the soft priority block."""

    priority: PriorityHint


class ContextEdit(BaseModel):
    type: Literal["offload", "prefetch", "evict"]
    # v1 only allows `session` (protocol §5.3); message/tool ranges are later.
    target: Literal["session"] = "session"


class ContextManagement(BaseModel):
    """The manage-request block (protocol §5.3).

    Present **only** on manage requests: `manage_request` is the JSON boolean
    ``true`` and v1 carries exactly one explicit edit. Ordinary requests must
    omit the whole block, so this plugin never sets it on the wire.
    """

    manage_request: Literal[True] = True
    edits: list[ContextEdit] = Field(..., min_length=1, max_length=1)


#: Fixed verb -> edit mapping (protocol §8.2).
MANAGE_EDITS = {
    "pause": "offload",
    "resume": "prefetch",
    "stop": "evict",
    "compact": "evict",
}


class HintRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    unit: Optional[Literal["message", "token"]] = None


class CacheControlHint(BaseModel):
    """One retention declaration (single object, never an array)."""

    type: Literal["ephemeral"]
    ttl_s: Optional[int] = None
    # 1-based effect point; equals `range.end`.
    msg_offset: Optional[int] = None
    range: Optional[HintRange] = None

    @model_validator(mode="after")
    def _offset_matches_range(self) -> "CacheControlHint":
        if (
            self.msg_offset is not None
            and self.range is not None
            and self.range.end is not None
            and self.msg_offset != self.range.end
        ):
            raise ValueError("cache_control.msg_offset must equal range.end")
        return self


class NextActionHint(BaseModel):
    state: Literal["running", "pause", "resume", "finalize"]
    tools: Optional[list[HintPendingTool]] = None
    prob: Optional[float] = None


class PredictionHint(BaseModel):
    latency_ms: Optional[int] = None
    remaining_calls: Optional[int] = None
    output_tokens: Optional[int] = None
    interval_ms: Optional[int] = None
    remaining_time_ms: Optional[int] = None
    expected_resume_s: Optional[int] = None
    next_action: Optional[NextActionHint] = None


class AgentHint(BaseModel):
    session: SessionHint
    # Manage requests only; ordinary requests omit the block (protocol §5.3).
    context_management: Optional[ContextManagement] = None
    agent_status: Optional[AgentStatusHint] = None
    priority: Optional[PriorityHint] = None
    priority_control: Optional[PriorityControl] = None
    strict_priority: Optional[PriorityHint] = None
    cache_control: Optional[CacheControlHint] = None
    prediction: Optional[PredictionHint] = None


def build_manage_hint(
    *,
    session_id: str,
    state: Literal["pause", "resume", "stop", "compact"],
    call_index: Optional[int] = None,
    epoch: Optional[int] = None,
    parent_session_id: Optional[str] = None,
) -> AgentHint:
    """Build one manage hint (session + context_management only).

    ``epoch`` is the epoch the edit applies to; a ``compact`` manage request
    carries the **old** epoch (protocol §6.3). This plugin has no lifecycle
    trigger to send one, but the builder pins the wire shape for consumers and
    for the shared golden vectors.
    """
    return AgentHint(
        session=SessionHint(
            session_id=session_id,
            parent_session_id=parent_session_id,
            call_index=call_index,
            epoch=epoch,
            state=state,
        ),
        context_management=ContextManagement(
            edits=[ContextEdit(type=MANAGE_EDITS[state])]
        ),
    )


def _wire_dict(hint: AgentHint) -> dict:
    """`exclude_none` dump with `session.parent_session_id` always present.

    ``exclude_none`` would drop the key on root sessions, but the contract
    requires it to exist with value ``null`` (no parent). Re-inserting it keeps
    the field order stable while preserving the "absent means unknown" rule for
    every other optional field.
    """
    data = hint.model_dump(mode="json", exclude_none=True)
    session = data.get("session") or {}
    ordered = {
        "session_id": session.get("session_id"),
        "parent_session_id": hint.session.parent_session_id,
    }
    ordered.update({key: value for key, value in session.items() if key != "session_id"})
    data["session"] = ordered
    return data


def encode_agent_hint(hint: AgentHint) -> str:
    """Serialize into the X-Agent-Hint header value (base64url, unpadded)."""
    payload = json.dumps(
        _wire_dict(hint), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def decode_agent_hint(value: str) -> AgentHint:
    padded = value + "=" * (-len(value) % 4)
    payload = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    return AgentHint.model_validate_json(payload)
