"""cache_control derivation for the Dify plugin (harness-side rules).

Implements ``CACHE_CONTROL_IMPLEMENTATION_GUIDE.md`` for the Dify model
provider: one ``ephemeral`` declaration with an explicit TTL, logical input
count as ``range.end``, a 128-token floor, and "don't declare" rules for a
rewritten prefix. The plugin is stateless across requests except for the
in-process prefix fingerprint table (a compaction signal), which is advisory:
a restart only costs one extra declaration.

The TTL gap comes from the merged hint data (the hint service is Dify's
observation channel): the predicted next tool duration, else the measured
inter-call interval, else the 300 s fallback.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Optional

MIN_CACHE_TOKENS = 128
TTL_MIN_S = 60
TTL_MAX_S = 3600
FALLBACK_GAP_S = 300.0
TTL_FACTOR = 2.0

_MAX_TRACKED_SESSIONS = 4096
_LOCK = threading.Lock()
_PREFIX_FINGERPRINTS: dict[str, tuple[int, str]] = {}


def estimate_ttl_s(
    *,
    tool_expected_s: Optional[float] = None,
    interval_ms: Optional[int] = None,
) -> int:
    """``clamp(gap * 2, 60, 3600)`` with the guide's gap priority order."""
    if tool_expected_s is not None and tool_expected_s > 0:
        gap = float(tool_expected_s)
    elif interval_ms is not None and interval_ms > 0:
        gap = float(interval_ms) / 1000.0
    else:
        gap = FALLBACK_GAP_S
    return max(TTL_MIN_S, min(TTL_MAX_S, int(round(gap * TTL_FACTOR))))


def estimate_tokens(value: Any) -> int:
    """Tokenizer-free estimate: CJK 1 token/char, other 1 token/4 chars."""
    cjk = 0
    other = 0

    def walk(item: Any) -> None:
        nonlocal cjk, other
        if item is None:
            return
        if isinstance(item, str):
            for char in item:
                code = ord(char)
                if (
                    0x2E80 <= code <= 0x9FFF
                    or 0xF900 <= code <= 0xFAFF
                    or 0xFF00 <= code <= 0xFFEF
                ):
                    cjk += 1
                else:
                    other += 1
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif hasattr(item, "content"):
            walk(getattr(item, "content"))
        elif hasattr(item, "text"):
            walk(getattr(item, "text"))

    walk(value)
    return cjk + (other + 3) // 4


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    value = getattr(role, "value", None)
    return str(value if value is not None else role or "").lower()


def _content(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def logical_message_count(messages: list[Any]) -> int:
    """Logical input count: every message, minus a trailing empty assistant."""
    kept = list(messages or [])
    if kept:
        last = kept[-1]
        if _role(last) == "assistant" and not _content(last).strip():
            kept = kept[:-1]
    return len(kept)


def _fingerprint(messages: list[Any]) -> tuple[int, str]:
    if not messages:
        return (0, "")
    first = _content(messages[0])
    return (len(messages), hashlib.sha256(first.encode("utf-8")).hexdigest()[:16])


def prefix_rewritten(prefix_key: str, messages: list[Any]) -> bool:
    """Record a cache prefix's fingerprint; True when it was rewritten.

    ``prefix_key`` is the content-derived prefix id (the node/system template
    hash), *not* the run session: different workflow nodes have different
    system prefixes and must not look like a compaction. For one prefix, a
    rewrite (count shrink or first-message change) means the previous context
    is gone, so the guide says "don't declare this round".
    """
    if not prefix_key:
        return False
    fingerprint = _fingerprint(messages)
    with _LOCK:
        previous = _PREFIX_FINGERPRINTS.get(prefix_key)
        if len(_PREFIX_FINGERPRINTS) >= _MAX_TRACKED_SESSIONS:
            _PREFIX_FINGERPRINTS.clear()
        _PREFIX_FINGERPRINTS[prefix_key] = fingerprint
    return previous is not None and (
        fingerprint[0] < previous[0] or fingerprint[1] != previous[1]
    )


def reset_prefix_tracking() -> None:
    """Drop the in-process fingerprint table (tests / tenant switch)."""
    with _LOCK:
        _PREFIX_FINGERPRINTS.clear()


def build_cache_control(
    *,
    logical_count: int,
    prefix_tokens: int,
    ttl_s: int,
) -> Optional[dict[str, Any]]:
    """One ``ephemeral`` declaration dict, or None when the guide says "don't"."""
    if not isinstance(logical_count, int) or logical_count < 1:
        return None
    if prefix_tokens < MIN_CACHE_TOKENS:
        return None
    if not isinstance(ttl_s, int) or ttl_s <= 0:
        return None
    return {
        "type": "ephemeral",
        "ttl_s": max(TTL_MIN_S, min(TTL_MAX_S, ttl_s)),
        "range": {"end": logical_count, "unit": "message"},
    }


def derive_cache_control(messages: list[Any], merged: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Full derivation from the prompt + merged prediction (guide §2/§4)."""
    logical = logical_message_count(messages)
    tokens = estimate_tokens([_content(message) for message in messages])
    prediction = merged.get("prediction") or {}
    if not isinstance(prediction, dict):
        prediction = {}
    next_action = prediction.get("next_action") or {}
    tools = next_action.get("tools") if isinstance(next_action, dict) else None
    tool_expected = None
    if isinstance(tools, list) and tools and isinstance(tools[0], dict):
        tool_expected = tools[0].get("expected_duration_s")
    ttl = estimate_ttl_s(
        tool_expected_s=tool_expected,
        interval_ms=prediction.get("interval_ms"),
    )
    return build_cache_control(
        logical_count=logical,
        prefix_tokens=tokens,
        ttl_s=ttl,
    )
