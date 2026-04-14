"""Run the Timbal HTTP server: ``python -m timbal.server``.

Optional leading ``serve`` is accepted so ``python -m timbal.server serve --port 4444`` works.
Voice UI: ``GET /voice`` and ``WebSocket /voice/ws`` for the loaded runnable (same app as ``/run``).
"""

from __future__ import annotations

import sys


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        argv = argv[1:]

    from .http import run_server_cli

    run_server_cli(argv)


if __name__ == "__main__":
    main()
