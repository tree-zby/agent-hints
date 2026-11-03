"""Agent hints data pipeline: four-tier merge -> AgentHint -> dual wire formats.

Merge order (low -> high precedence; higher tiers override per-field):
  1. auto defaults        (identity + priority level 3 / user_waiting)
  2. credential static    (hints_json: partial AgentHint JSON)
  3. node parameters      (priority / osl / iat completion_params)
  4. hint service         (GET hint_service_url at request time; silent degradation)

Output priority level is folded into the two-value domain {1, 3} (v2 contract:
3 = user-intervened / admission-exempt, 1 = everything else; see
normalize_priority_level).

Outputs:
  - AgentHint object -> X-Agent-Hint header (UTF-8 JSON -> base64url), full
    agreed schema (see docs/SCHEMA.md)
  - nvext.agent_hints request-body projection (NVIDIA Dynamo v1.5.1:
    priority only, higher values have higher priority)

Identity (session.session_id): `user` invocation param when present, else the
content-derived prefix hash, else a credential-provided static id. The prefix
hash (sha256 over the first system message, hex[:16]) is also used as
prefix_id in hint service queries. It is not sent in nvext.agent_hints.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from .cache_policy import (
    build_cache_control,
    derive_cache_control,
    estimate_tokens,
    logical_message_count,
    prefix_rewritten,
)
from .hint import AgentHint

FetchFn = Callable[[str, float], Optional[dict]]


_WIRE_STATES = frozenset({"running", "pause", "resume", "finalize"})
_STATE_FOLDS = {"tool": "running", "suspend": "pause"}


def fold_state(value: Any, *, default: str = "running") -> str:
    """Fold an internal/legacy state into the 4-value wire contract.

    ``tool`` -> ``running``, ``suspend`` -> ``pause`` (AGENT_STATUS_STATE_DESIGN
    v1.0); unknown values degrade to ``default`` so a producer's vocabulary
    change never invalidates the hint.
    """
    state = str(value or "").strip().lower()
    state = _STATE_FOLDS.get(state, state)
    return state if state in _WIRE_STATES else default


def normalize_priority_level(level: int) -> int:
    """Fold a priority level into the two-value domain {1, 3}.

    v2 cross-harness contract: 3 = user-intervened (admission-exempt),
    1 = everything else. The numbers are contract anchors (the Motor
    admission line is 3), not tiers. v1 legacy {2, 4} map to the nearest
    value; any other integer folds by the admission line (>=3 -> 3,
    else -> 1) — admission outcomes are unchanged.
    """
    if level in (1, 3):
        return level
    if level == 2:
        return 1
    if level == 4:
        return 3
    return 3 if level >= 3 else 1


def deep_merge(base: dict, override: dict) -> dict:
    """Nested dict merge; override wins per key, lists/scalars replace."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _message_role(msg: Any) -> str:
    role = getattr(msg, "role", None)
    value = getattr(role, "value", None)
    return str(value if value is not None else role)


def _message_content(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            data = getattr(item, "data", None)
            if isinstance(data, str):
                parts.append(data)
        return "".join(parts)
    return str(content)


def prefix_hash(prompt_messages: Any) -> str:
    """sha256 of the first non-empty system message content, hex[:16]."""
    fallback = ""
    for msg in prompt_messages or []:
        content = _message_content(msg)
        if not content:
            continue
        if _message_role(msg) == "system":
            return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        if not fallback:
            fallback = content
    if fallback:
        return hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:16]
    return "adhoc"


def _parse_hints_json(raw: Any) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def http_fetch(url: str, timeout: float) -> Optional[dict]:
    """Default FetchFn: GET url, return parsed JSON dict or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            parsed = json.loads(resp.read().decode("utf-8"))
            return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001 — hint service outages must degrade silently
        return None


def service_query_url(base_url: str, *, prefix_id: str, user: str, model: str) -> str:
    query = urllib.parse.urlencode({"prefix_id": prefix_id, "user": user, "model": model})
    return f"{base_url.rstrip('/')}/v1/hints?{query}"


def _auto_pin_ttl_s(raw: Any) -> Optional[int]:
    """Positive integer TTL for the auto pin; invalid input degrades to a ttl-less pin."""
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def build_hint(
    prompt_messages: Any,
    model_parameters: dict,
    credentials: dict,
    user: Optional[str] = None,
    model: str = "",
    fetch_fn: Optional[FetchFn] = None,
) -> tuple[AgentHint, dict]:
    """Run the four-tier pipeline; returns (agent_hint, nvext_agent_hints).

    Never raises: every stage degrades silently (bad JSON / absent service /
    malformed params simply contribute nothing).
    """
    prompt_messages = prompt_messages or []
    prefix_id = prefix_hash(prompt_messages)
    now_ms = int(time.time() * 1000)

    # tier 1: auto defaults
    identity_mode = (credentials.get("hint_identity") or "auto").strip()
    if identity_mode == "user":
        session_id = (user or "").strip() or prefix_id
    elif identity_mode == "static":
        static = _parse_hints_json(credentials.get("hints_json"))
        session_id = str(
            ((static.get("session") or {}).get("session_id")) or prefix_id
        )
    else:  # auto
        session_id = (user or "").strip() or prefix_id

    # tier-1 baseline: derived from the request itself (the plugin's own
    # capability boundary). A batch deployment runs without a waiting user
    # (`run_mode=batch` credential); otherwise a continuation whose last input
    # is a tool result / assistant turn is self-driven (`critical_path`),
    # while a user turn is user-intervened (`user_waiting`). The two-value
    # domain is enforced at the exit; explicit node/credential/service values
    # still override per tier precedence.
    run_mode = (credentials.get("run_mode") or "interactive").strip().lower()
    if run_mode == "batch":
        baseline_priority = {"level": 1, "reason": "background"}
    elif (_message_role(prompt_messages[-1]).lower() if prompt_messages else "") in (
        "tool",
        "assistant",
    ):
        baseline_priority = {"level": 1, "reason": "critical_path"}
    else:
        baseline_priority = {"level": 3, "reason": "user_waiting"}

    merged: dict = {
        "session": {"session_id": session_id},
        "priority": baseline_priority,
    }

    # tier 2: credential static hints_json (partial AgentHint)
    deep_merge(merged, _parse_hints_json(credentials.get("hints_json")))

    # tier 3: node completion_params (priority plus header prediction metadata)
    node_priority = _as_int(model_parameters.get("priority"))
    if node_priority is not None:
        merged.setdefault("priority", {})["level"] = node_priority
    node_osl = _as_int(model_parameters.get("osl"))
    node_iat = _as_int(model_parameters.get("iat"))
    if node_osl is not None:
        merged.setdefault("prediction", {})["output_tokens"] = node_osl
    if node_iat is not None:
        merged.setdefault("prediction", {})["interval_ms"] = node_iat

    # tier 4: hint service (dynamic enrichment; silent degradation)
    service_url = (credentials.get("hint_service_url") or "").strip()
    if service_url and fetch_fn is None:
        fetch_fn = http_fetch
    if service_url and fetch_fn is not None:
        try:
            timeout = _as_float(credentials.get("hint_service_timeout")) or 2.0
        except Exception:  # noqa: BLE001
            timeout = 2.0
        try:
            dynamic = fetch_fn(
                service_query_url(service_url, prefix_id=prefix_id,
                                  user=user or "", model=model),
                timeout,
            )
        except Exception:  # noqa: BLE001 — hint service outages must degrade silently
            dynamic = None
        if dynamic:
            deep_merge(merged, dynamic)

    # agent_status: the in-band lifecycle snapshot (protocol §4). The plugin
    # only runs while a model request is being prepared, so the real state is
    # ``running``; a run-scoped hint-service binding may override the state and
    # state_since_ms with real run facts.
    merged.setdefault(
        "agent_status", {"state": "running", "state_since_ms": now_ms}
    )
    # 4-value wire contract: fold internal states from any producer at the
    # boundary (tool -> running, suspend -> pause; unknown -> running).
    status_block = merged.get("agent_status")
    if isinstance(status_block, dict):
        status_block["state"] = fold_state(status_block.get("state"))
    prediction_block = merged.get("prediction")
    if isinstance(prediction_block, dict):
        next_action_block = prediction_block.get("next_action")
        if isinstance(next_action_block, dict) and next_action_block.get("state") is not None:
            next_action_block["state"] = fold_state(next_action_block.get("state"))

    # cache_control (opt-out via credential cache_pin=false), derived per the
    # cache-control guide: one ``ephemeral`` declaration with an explicit TTL.
    # The gap comes from the merged prediction (next tool expected duration ->
    # measured inter-call interval -> 300 s fallback); ``range.end`` is the
    # logical input count; nothing is declared below one 128-token block or on
    # a rewritten prefix (compaction/branch). An explicit cache_pin_ttl_s
    # overrides the derived TTL.
    if credentials.get("cache_pin", "true").strip().lower() != "false" \
            and "cache_control" not in merged:
        # Cache prefixes belong to the node/system template, not the run: two
        # different workflow nodes naturally have different system prefixes and
        # must not be mistaken for a compaction rewrite. Key the fingerprint by
        # the content-derived prefix id (same value the hint service uses).
        if not prefix_rewritten(prefix_id, prompt_messages):
            ttl_override = _auto_pin_ttl_s(credentials.get("cache_pin_ttl_s"))
            declaration = (
                build_cache_control(
                    logical_count=logical_message_count(prompt_messages),
                    prefix_tokens=estimate_tokens(
                        [_message_content(message) for message in prompt_messages]
                    ),
                    ttl_s=ttl_override,
                )
                if ttl_override is not None
                else derive_cache_control(prompt_messages, merged)
            )
            if declaration is not None:
                merged["cache_control"] = [declaration]

    # v2 two-value fold: priority.level ∈ {1, 3} (3 = user-intervened).
    # Applied after all tiers so legacy {2, 4} and out-of-domain node params
    # or service values narrow without changing admission outcomes.
    priority_block = merged.get("priority")
    if isinstance(priority_block, dict):
        level = priority_block.get("level")
        if isinstance(level, int) and not isinstance(level, bool):
            priority_block["level"] = normalize_priority_level(level)

    hint = AgentHint.model_validate(merged)

    # Dynamo v1.5.1 rejects unknown AgentHints fields. Send only priority;
    # keep prediction metadata and session identity in X-Agent-Hint.
    level = hint.priority.level if hint.priority is not None else 1
    return hint, {"priority": int(level)}
