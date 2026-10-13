"""hint_service — 调度侧数据中枢（独立包，零第三方依赖）。"""
from .app import serve
from .store import HintStore

__all__ = ["HintStore", "serve"]
