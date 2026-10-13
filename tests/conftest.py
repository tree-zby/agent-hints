"""Shared test setup: dify_plugin stubs + module loading for models/llm."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_LLM_DIR = REPO_ROOT / "models" / "llm"


def _install_dify_plugin_stubs() -> None:
    if "dify_plugin" in sys.modules:
        return
    dp = types.ModuleType("dify_plugin")
    dp.ModelProvider = object

    entities = types.ModuleType("dify_plugin.entities")
    model_mod = types.ModuleType("dify_plugin.entities.model")
    for name in ("AIModelEntity", "FetchFrom", "ModelPropertyKey", "ModelType", "ParameterRule"):
        setattr(model_mod, name, type(name, (), {}))
    message_mod = types.ModuleType("dify_plugin.entities.model.message")
    message_mod.AssistantPromptMessage = type("AssistantPromptMessage", (), {})
    llm_entities = types.ModuleType("dify_plugin.entities.model.llm")
    for name in ("LLMResult", "LLMResultChunk", "LLMResultChunkDelta", "LLMUsage"):
        setattr(llm_entities, name, type(name, (), {}))

    errors = types.ModuleType("dify_plugin.errors")
    errors_model = types.ModuleType("dify_plugin.errors.model")
    errors_model.CredentialsValidateFailedError = type(
        "CredentialsValidateFailedError", (Exception,), {}
    )

    interfaces = types.ModuleType("dify_plugin.interfaces")
    interfaces_model = types.ModuleType("dify_plugin.interfaces.model")
    large_language_model = types.ModuleType("dify_plugin.interfaces.model.large_language_model")
    large_language_model.LargeLanguageModel = object

    dp.entities = entities
    entities.model = model_mod
    model_mod.llm = llm_entities
    model_mod.message = message_mod
    dp.errors = errors
    errors.model = errors_model
    dp.interfaces = interfaces
    interfaces.model = interfaces_model
    interfaces_model.large_language_model = large_language_model

    for module in (dp, entities, model_mod, message_mod, llm_entities, errors,
                   errors_model, interfaces, interfaces_model, large_language_model):
        sys.modules[module.__name__] = module


_install_dify_plugin_stubs()
_pkg = types.ModuleType("agent_hints_pkg")
_pkg.__path__ = [str(_LLM_DIR)]
sys.modules["agent_hints_pkg"] = _pkg
_llm_spec = importlib.util.spec_from_file_location("agent_hints_pkg.llm", _LLM_DIR / "llm.py")
llm_mod = importlib.util.module_from_spec(_llm_spec)
sys.modules["agent_hints_pkg.llm"] = llm_mod
_llm_spec.loader.exec_module(llm_mod)

hint_mod = importlib.import_module("agent_hints_pkg.hint")
pipeline_mod = importlib.import_module("agent_hints_pkg.pipeline")
