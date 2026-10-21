"""Load the installed Dify model SDK without starting the plugin runtime."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_LLM_DIR = REPO_ROOT / "models" / "llm"


def _load_dify_model_sdk() -> None:
    if "dify_plugin" in sys.modules:
        return
    spec = importlib.util.find_spec("dify_plugin")
    if spec is None or spec.origin is None:
        raise ImportError("Tests require the installed dify_plugin SDK")
    sdk_root = Path(spec.origin).parent
    # Skip package initializers that load plugin configuration or patch sockets.
    # Entities, parameter validation and LargeLanguageModel.invoke stay real.
    for suffix in ("", ".interfaces", ".interfaces.model"):
        name = "dify_plugin" + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(sdk_root.joinpath(*suffix.strip(".").split(".")))]
        sys.modules[name] = module
        if suffix:
            parent, _, child = name.rpartition(".")
            setattr(sys.modules[parent], child, module)


_load_dify_model_sdk()
_pkg = types.ModuleType("agent_hints_pkg")
_pkg.__path__ = [str(_LLM_DIR)]
sys.modules["agent_hints_pkg"] = _pkg
_llm_spec = importlib.util.spec_from_file_location("agent_hints_pkg.llm", _LLM_DIR / "llm.py")
llm_mod = importlib.util.module_from_spec(_llm_spec)
sys.modules["agent_hints_pkg.llm"] = llm_mod
_llm_spec.loader.exec_module(llm_mod)

hint_mod = importlib.import_module("agent_hints_pkg.hint")
pipeline_mod = importlib.import_module("agent_hints_pkg.pipeline")
