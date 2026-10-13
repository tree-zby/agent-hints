"""Agent Hints — OpenAI-compatible model provider with agentic inference hints.

Standalone Dify model plugin. Attaches scheduling hints to every model
request so hint-aware inference gateways (e.g. NVIDIA Dynamo) can do
priority routing, KV-cache affinity and capacity planning:

- nvext.agent_hints (request body, Dynamo-compatible projection; default on)
- X-Agent-Hint header (full agreed schema, see docs/SCHEMA.md; optional)

Data sources (four tiers, low -> high precedence):
  1. auto defaults        (identity + priority level 1)
  2. credential hints_json (static partial AgentHint)
  3. node completion_params (priority / osl / iat / latency_sensitivity)
  4. hint service          (GET hint_service_url at request time; the service
     may return any partial AgentHint — session enrichment, prediction stats,
     next_action — which is merged verbatim into the final object)

No external tooling is required: with zero configuration the plugin emits
safe defaults and degrades to a plain request if anything fails.
"""
from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from decimal import Decimal
from typing import Any, Optional, Union

import httpx
from dify_plugin import ModelProvider
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import AssistantPromptMessage
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

from .hint import AGENT_HINT_HEADER, encode_agent_hint
from .pipeline import build_hint

# ---------------------------------------------------------------------------
# messages / body
# ---------------------------------------------------------------------------
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


def normalize_messages(prompt_messages: Sequence) -> list[dict]:
    """Normalize messages into plain dicts (no content rewriting — the outbound prompt stays untouched)."""
    return [
        {"role": _message_role(msg), "content": _message_content(msg)}
        for msg in prompt_messages
    ]


def parse_extra_headers(raw: str) -> dict[str, str]:
    """Credential extra_headers_json -> pool-level static headers; invalid input is silently ignored."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if v is not None}


def build_headers(credentials: dict, hint) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {credentials.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    headers.update(parse_extra_headers(credentials.get("extra_headers_json", "")))
    if credentials.get("emit_header", "true").strip().lower() != "false":
        headers[AGENT_HINT_HEADER] = encode_agent_hint(hint)
    return headers


def build_body(
    model: str,
    messages: list[dict],
    model_parameters: dict,
    stop: Optional[Sequence[str]],
    stream: bool,
    nvext: Optional[dict] = None,
    emit_nvext: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    for key in ("temperature", "max_tokens", "top_p"):
        val = model_parameters.get(key)
        if val is not None:
            body[key] = val
    if nvext and emit_nvext:
        body["nvext"] = {"agent_hints": nvext}
    if stop:
        body["stop"] = list(stop)
    return body


# ---------------------------------------------------------------------------
# LLM implementation
# ---------------------------------------------------------------------------
class AgentHintsModel(LargeLanguageModel):
    """OpenAI-compatible LLM with Agent Hints injection."""

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: Sequence,
        model_parameters: dict,
        tools: Optional[Sequence] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        endpoint_url = credentials.get("endpoint_url", "").rstrip("/")

        messages = normalize_messages(prompt_messages)
        hint, nvext = build_hint(prompt_messages, model_parameters, credentials,
                                 user=user, model=model)

        headers = build_headers(credentials, hint)
        emit_nvext = credentials.get("emit_nvext", "true").strip().lower() != "false"
        body = build_body(model, messages, model_parameters, stop, stream, nvext,
                          emit_nvext=emit_nvext)

        url = f"{endpoint_url}/chat/completions"
        if stream:
            return self._stream_invoke(url, headers, body, model)
        return self._block_invoke(url, headers, body, model)

    def _stream_invoke(self, url, headers, body, model) -> Generator[LLMResultChunk, None, None]:
        with httpx.Client(timeout=300) as client:
            with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    err = resp.read().decode(errors="replace")[:300]
                    raise RuntimeError(f"inference endpoint {resp.status_code}: {err}")
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    finish = choice.get("finish_reason")
                    usage = self._usage(chunk.get("usage") or {}) if finish else None
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=[],
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=delta.get("content", "")),
                            usage=usage,
                            finish_reason=finish,
                        ),
                    )
                    if finish:
                        break

    def _block_invoke(self, url, headers, body, model) -> LLMResult:
        with httpx.Client(timeout=300) as client:
            resp = client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"inference endpoint {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return LLMResult(
                model=model,
                prompt_messages=[],
                message=AssistantPromptMessage(content=msg.get("content", "")),
                usage=self._usage(data.get("usage") or {}),
            )

    @staticmethod
    def _usage(raw: dict) -> LLMUsage:
        return LLMUsage(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
            prompt_unit_price=Decimal("0.0"),
            prompt_price_unit=Decimal("0.0"),
            prompt_price=Decimal("0.0"),
            completion_unit_price=Decimal("0.0"),
            completion_price_unit=Decimal("0.0"),
            completion_price=Decimal("0.0"),
            total_price=Decimal("0.0"),
            currency="USD",
            latency=0.0,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list,
        tools: Optional[list] = None,
    ) -> int:
        return sum(len(str(m.get("content") or "")) // 2 for m in normalize_messages(prompt_messages))

    def validate_credentials(self, model: str, credentials: dict) -> None:
        endpoint_url = credentials.get("endpoint_url", "").rstrip("/")
        api_key = credentials.get("api_key", "")
        url = f"{endpoint_url}/models"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code != 200:
                    raise CredentialsValidateFailedError(
                        f"endpoint validation failed: {resp.status_code}"
                    )
        except httpx.ConnectError as e:
            raise CredentialsValidateFailedError(f"endpoint unreachable: {e}") from e

    @property
    def _invoke_error_mapping(self) -> dict:
        return {}

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        def _rule(variable, label, ptype, default, help_text, vmin=None, vmax=None):
            rule = ParameterRule(
                name=variable,
                variable=variable,
                label={"en_US": label, "zh_Hans": label},
                type=ptype,
                help={"en_US": help_text, "zh_Hans": help_text},
                required=False,
                default=default,
            )
            if vmin is not None:
                rule.min = vmin
            if vmax is not None:
                rule.max = vmax
            return rule

        return AIModelEntity(
            model=model,
            label={"en_US": model, "zh_Hans": model},
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: credentials.get("mode", "chat"),
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", "4096")),
            },
            parameter_rules=[
                _rule("priority", "Scheduling Priority", "int", 1,
                      "Higher = more urgent. Converted to Dynamo's inverted "
                      "priority scale in nvext.agent_hints.", 0, 100),
                _rule("osl", "Expected Output Tokens (osl)", "int", 512,
                      "Expected output sequence length for capacity planning.", 1, 100000),
                _rule("iat", "Expected Inter-arrival (ms)", "int", 250,
                      "Expected request inter-arrival time for load anticipation.",
                      0, 600000),
                _rule("latency_sensitivity", "Latency Sensitivity", "float", 1.0,
                      "Higher values get priority routing.", 0.0, 100.0),
            ],
        )
