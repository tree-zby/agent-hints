"""Agent Hint schema — field-for-field mirror of the agreed cross-harness
specification.

Field naming and wire encoding are pinned by a cross-language golden vector
in tests/test_schema_golden.py. Wire form: the AgentHint object is serialized
as UTF-8 JSON and encoded as base64url into the `X-Agent-Hint` HTTP header;
the model request body keeps its original format (nvext.agent_hints is a
separate, Dynamo-compatible projection produced by the pipeline, not part of
this schema).

Four blocks are required on every request (missing = non-compliant; extra
fields are allowed and ignored):

  session             identity + per-request lifecycle summary
  context_management  whether this is a management request, plus context edits
  priority_control    canonical location of the soft priority block
  cache_control       one ephemeral retention declaration, never an array

`priority` (top-level) stays as a compatibility mirror of
`priority_control.priority`, and `agent_status` keeps carrying the 4-value
lifecycle word used by the event channel.

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
    target: Literal["session", "messages", "tools"] = "session"


class ContextManagement(BaseModel):
    """Whether this is a pure context-management request, plus its edits.

    This plugin only builds requests that the agent was going to send anyway,
    so `manage_request` is always the string ``"false"``; the edits describe
    context intent attached to the request.
    """

    manage_request: Literal["true", "false"] = "false"
    edits: list[ContextEdit] = Field(default_factory=list)


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
    context_management: ContextManagement
    agent_status: Optional[AgentStatusHint] = None
    priority: Optional[PriorityHint] = None
    priority_control: Optional[PriorityControl] = None
    strict_priority: Optional[PriorityHint] = None
    cache_control: Optional[CacheControlHint] = None
    prediction: Optional[PredictionHint] = None


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
