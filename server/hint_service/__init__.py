"""hint_service — the data plane of the scheduling side (standalone, zero dependencies)."""
from .app import serve
from .store import HintStore

__all__ = ["HintStore", "serve"]
