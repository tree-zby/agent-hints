"""python -m hint_service — start the hint service."""
from __future__ import annotations

import argparse

from .app import serve
from .store import HintStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hint_service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--store", default="", help="JSON persistence path (empty = in-memory)")
    args = parser.parse_args(argv)

    store = HintStore(args.store) if args.store else HintStore()
    server = serve(args.host, args.port, store)
    print(f"hint-service listening on {args.host}:{args.port} "
          f"(store={'file' if args.store else 'memory'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
