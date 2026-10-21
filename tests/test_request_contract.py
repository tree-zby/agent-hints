"""Dynamo v1.5.1 contract checks against serialized HTTP request bodies.

Run this module directly to print a report from the same request captures.
"""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from tests.conftest import hint_mod, llm_mod, pipeline_mod
from dify_plugin.entities.model import ParameterType
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage


# AgentHints in Dynamo v1.5.1 denies unknown fields.
DYNAMO_AGENT_HINT_FIELDS = {
    "priority", "strict_priority", "osl", "speculative_prefill", "latency_sensitivity",
}
PRIORITY_CASES = [(0, 0), (5, 5), (10, 10), (None, 1)]
GENERATION_PARAMETERS = {"temperature", "max_tokens", "top_p"}
MODEL = "contract-test-model"
MESSAGES = [
    SystemPromptMessage(content="You are a helpful assistant."),
    UserPromptMessage(content="Say hello."),
]


def capture_request(monkeypatch, parameters=None, *, credentials=None, stream=False):
    """Use the public SDK invoke path and capture bytes at the HTTP transport."""
    requests = []
    client_class = httpx.Client

    def handle(request):
        requests.append(request)
        if stream:
            chunks = [
                {"choices": [{"delta": {"content": "Hello."}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(200, text=text + "data: [DONE]\n\n",
                                  headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Hello."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        })

    def mock_client(**kwargs):
        return client_class(transport=httpx.MockTransport(handle), trust_env=False, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(llm_mod.httpx, "Client", mock_client)
        model = llm_mod.AgentHintsModel(model_schemas=[])
        results = list(model.invoke(
            model=MODEL,
            credentials={"endpoint_url": "http://dynamo.invalid/v1", **(credentials or {})},
            prompt_messages=MESSAGES,
            model_parameters=parameters or {},
            stream=stream,
        ))
    assert "".join(result.delta.message.content for result in results) == "Hello."
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://dynamo.invalid/v1/chat/completions"
    return requests[0]


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize("node_priority, expected", PRIORITY_CASES)
def test_request_priority_contract(monkeypatch, node_priority, expected, stream):
    params = {} if node_priority is None else {"priority": node_priority}
    request = capture_request(monkeypatch, params, stream=stream)
    body = json.loads(request.content)
    hints = body["nvext"]["agent_hints"]
    assert set(hints) == {"priority"}
    assert set(hints) <= DYNAMO_AGENT_HINT_FIELDS
    assert "prefix_id" not in hints and "iat" not in hints
    assert type(hints["priority"]) is int
    assert hints["priority"] == expected
    assert body["model"] == MODEL
    assert body["stream"] is stream
    assert body["messages"] == llm_mod.normalize_messages(MESSAGES)
    assert GENERATION_PARAMETERS.isdisjoint(body)


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize("params", [
    {},
    {"temperature": None, "max_tokens": None, "top_p": None},
    {"temperature": 0},
    {"max_tokens": 64},
    {"top_p": 0.8},
    {"temperature": 0.25, "max_tokens": 64, "top_p": 0.8},
])
def test_request_generation_parameters(monkeypatch, params, stream):
    body = json.loads(capture_request(monkeypatch, params, stream=stream).content)
    actual = {key: body[key] for key in GENERATION_PARAMETERS if key in body}
    assert actual == {key: value for key, value in params.items() if value is not None}


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
def test_request_emit_nvext_false(monkeypatch, stream):
    request = capture_request(monkeypatch, {"priority": 5, "temperature": 0.25},
                              credentials={"emit_nvext": "false"}, stream=stream)
    body = json.loads(request.content)
    assert "nvext" not in body
    assert body["temperature"] == 0.25
    assert hint_mod.decode_agent_hint(request.headers["X-Agent-Hint"]).priority.level == 5


def test_other_hints_stay_in_header_and_service_query(monkeypatch):
    queries = []

    def fetch(url, timeout):
        queries.append(parse_qs(urlsplit(url).query, keep_blank_values=True))
        return {
            "priority": {"level": 10},
            "prediction": {"output_tokens": 42, "interval_ms": 17},
            "strict_priority": {"level": 2},
            "prefix_id": "must-not-leak", "iat": 17,
            "osl": 42, "latency_sensitivity": 2.5, "unexpected": "ignored",
        }

    monkeypatch.setattr(pipeline_mod, "http_fetch", fetch)
    request = capture_request(
        monkeypatch, {"priority": 5, "osl": 128, "iat": 90, "latency_sensitivity": 9.0},
        credentials={
            "hint_service_url": "http://hint-service.invalid",
            "latency_sensitivity": 7.0,
            "hints_json": json.dumps({"priority": {"level": 3}, "prefix_id": "static"}),
        },
    )
    assert queries == [{"prefix_id": [pipeline_mod.prefix_hash(MESSAGES)],
                        "user": [""], "model": [MODEL]}]
    hint = hint_mod.decode_agent_hint(request.headers["X-Agent-Hint"])
    assert hint.session.session_id == pipeline_mod.prefix_hash(MESSAGES)
    assert hint.priority.level == 10
    assert hint.strict_priority.level == 2
    assert hint.prediction.output_tokens == 42
    assert hint.prediction.interval_ms == 17
    assert json.loads(request.content)["nvext"]["agent_hints"] == {"priority": 10}


def test_generation_parameter_schema_uses_real_sdk_rules():
    schema = llm_mod.AgentHintsModel(model_schemas=[]).get_customizable_model_schema(MODEL, {})
    rules = {rule.name: rule for rule in schema.parameter_rules}
    for name, ptype in [("temperature", ParameterType.FLOAT), ("max_tokens", ParameterType.INT),
                        ("top_p", ParameterType.FLOAT)]:
        rule = rules[name]
        assert rule.type == ptype
        assert not rule.required
        assert rule.min <= rule.default <= rule.max
    assert rules["priority"].help.zh_hans == "数值越大越优先，原样发送给 Dynamo"
    assert "latency_sensitivity" not in rules


def print_capture_report():
    """Render every value from decoded transport captures, not expected values."""
    with pytest.MonkeyPatch.context() as patch:
        print("| 节点值 | 实际请求 priority 值 | 实际 agent_hints 键 |")
        print("|---|---|---|")
        for node_priority, _ in PRIORITY_CASES:
            params = {} if node_priority is None else {"priority": node_priority}
            body = json.loads(capture_request(patch, params).content)
            hints = body["nvext"]["agent_hints"]
            label = "未设置（默认）" if node_priority is None else str(node_priority)
            keys = json.dumps(sorted(hints), ensure_ascii=False)
            print(f"| {label} | {hints['priority']} | `{keys}` |")
        request = capture_request(patch, {"priority": 5, "temperature": 0.25,
                                          "max_tokens": 64, "top_p": 0.8})
        print(json.dumps(json.loads(request.content), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print_capture_report()
