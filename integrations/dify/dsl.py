"""Dify app DSL -> node prefix / graph shape.

The plugin's join key is the content hash of the first system message
(``sha256(text)[:16]``). A driver that sees the workflow run but not the
model request can compute the same key from the app DSL, provided the system
template is static (a template with per-call variables renders differently and
will not join).

Requires PyYAML (optional dependency of the companion integration).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _graph(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    return (doc.get("workflow") or {}).get("graph") or doc


def load_dsl_info(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Return ``(node_prefixes, successors, node_types)`` for a Dify app DSL."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PyYAML is required for the Dify integration (pip install pyyaml)"
        ) from exc

    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    graph = _graph(doc)
    prefixes: dict[str, str] = {}
    node_types: dict[str, str] = {}
    successors: dict[str, list[str]] = {}

    for node in graph.get("nodes") or []:
        node_id = node.get("id")
        if not node_id:
            continue
        data = node.get("data") or {}
        node_type = str(data.get("type") or "")
        node_types[str(node_id)] = node_type
        if node_type != "llm":
            continue
        system = next(
            (
                message
                for message in data.get("prompt_template") or []
                if message.get("role") == "system"
                and isinstance(message.get("text"), str)
                and message["text"].strip()
            ),
            None,
        )
        if system is not None:
            prefixes[str(node_id)] = hashlib.sha256(
                system["text"].encode("utf-8")
            ).hexdigest()[:16]

    for edge in graph.get("edges") or []:
        source, target = edge.get("source"), edge.get("target")
        if source and target:
            successors.setdefault(str(source), []).append(str(target))

    return prefixes, successors, node_types


def reachable_llm_nodes(
    start: str,
    successors: dict[str, list[str]],
    node_types: dict[str, str],
) -> list[str]:
    """LLM node ids reachable after ``start``, in BFS order."""
    queue = list(successors.get(start, []))
    seen: set[str] = set()
    llm_nodes: list[str] = []
    while queue:
        node_id = queue.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_types.get(node_id) == "llm":
            llm_nodes.append(node_id)
        queue.extend(successors.get(node_id, []))
    return llm_nodes
