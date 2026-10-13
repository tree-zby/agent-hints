"""Agent Hint schema — field-for-field mirror of the agreed cross-harness
specification.

Field naming and wire encoding are pinned by a cross-language golden vector
in tests/test_schema_golden.py. Wire form: the AgentHint object is serialized
as UTF-8 JSON and encoded as base64url into the `X-Agent-Hint` HTTP header;
the model request body keeps its original format (nvext.agent_hints is a
separate, Dynamo-compatible projection produced by the pipeline, not part of
this schema).

See docs/SCHEMA.md for the full field reference.
"""
import base64
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

AGENT_HINT_HEADER = "X-Agent-Hint"


class HintToolRef(BaseModel):
    name: str
    call_id: Optional[str] = None
    child_session_id: Optional[str] = None


class HintPendingTool(BaseModel):
    name: str
    expected_duration_s: Optional[float] = None


class SessionHint(BaseModel):
    session_id: str
    parent_session_id: Optional[str] = None
    derivation: Optional[dict] = None
    agent_type: Optional[str] = None
    call_index: Optional[int] = None
    epoch: Optional[int] = None

    @field_validator("session_id")
    @classmethod
    def _session_id_required(cls, value: str) -> str:
        if not value:
            raise ValueError("session.session_id must be non-empty")
        return value


class AgentStatusHint(BaseModel):
    state: Literal["running", "tool", "pause", "suspend", "finalize"]
    state_since_ms: Optional[int] = None
    reason: Optional[str] = None
    tools: Optional[list[HintToolRef]] = None


class PriorityHint(BaseModel):
    level: int
    reason: Optional[str] = None


class HintRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    unit: Optional[Literal["message", "token"]] = None


class CacheControlHint(BaseModel):
    type: Literal["pin", "ephemeral"]
    ttl_s: Optional[int] = None
    range: Optional[HintRange] = None


class NextActionHint(BaseModel):
    state: Literal["running", "tool", "pause", "suspend", "finalize"]
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
    agent_status: Optional[AgentStatusHint] = None
    priority: Optional[PriorityHint] = None
    strict_priority: Optional[PriorityHint] = None
    cache_control: Optional[list[CacheControlHint] | CacheControlHint] = None
    prediction: Optional[PredictionHint] = None


def encode_agent_hint(hint: AgentHint) -> str:
    """Serialize into the X-Agent-Hint header value (base64url, unpadded)."""
    payload = hint.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def decode_agent_hint(value: str) -> AgentHint:
    padded = value + "=" * (-len(value) % 4)
    payload = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    return AgentHint.model_validate_json(payload)
