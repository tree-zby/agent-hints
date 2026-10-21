"""Agent hints data pipeline: four-tier merge -> AgentHint -> dual wire formats.

Merge order (low -> high precedence; higher tiers override per-field):
  1. auto defaults        (identity + priority level 1)
  2. credential static    (hints_json: partial AgentHint JSON)
  3. node parameters      (priority / osl / iat completion_params)
  4. hint service         (GET hint_service_url at request time; silent degradation)

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
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from .hint import AgentHint

FetchFn = Callable[[str, float], Optional[dict]]


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

    merged: dict = {
        "session": {"session_id": session_id},
        "priority": {"level": 1},
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

    # cache_control auto pin (opt-out via credential cache_pin=false)
    if credentials.get("cache_pin", "true").strip().lower() != "false" \
            and "cache_control" not in merged:
        merged["cache_control"] = [
            {"type": "pin", "range": {"end": 1, "unit": "message"}}
        ]

    hint = AgentHint.model_validate(merged)

    # Dynamo v1.5.1 rejects unknown AgentHints fields. Send only priority;
    # keep prediction metadata and session identity in X-Agent-Hint.
    level = hint.priority.level if hint.priority is not None else 1
    return hint, {"priority": int(level)}
