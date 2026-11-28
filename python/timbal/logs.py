import logging
import os
import sys
import threading
from typing import Any

import structlog

# Event names that can be suppressed via TIMBAL_SUPPRESS_EVENTS env var
# Format: comma-separated list of event names, e.g. "tracing_setup,parent_trace_not_found"
SUPPRESSED_EVENTS: set[str] = set()

# Set on the root handler `setup_logging()` adds, so a later call reconfigures it instead of adding another.
_HANDLER_MARK = "_timbal_setup_logging"

# `setup_logging()` runs one call at a time: two at once could each find no handler of timbal's and both add one.
_SETUP_LOCK = threading.Lock()


def _new_setup_lock() -> None:
    global _SETUP_LOCK
    _SETUP_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):  # (not on Windows)
    os.register_at_fork(after_in_child=_new_setup_lock)  # held by another thread at the fork, it would never be let go


def _load_suppressed_events() -> None:
    """Load suppressed events from environment variable."""
    global SUPPRESSED_EVENTS
    suppressed = os.getenv("TIMBAL_SUPPRESS_EVENTS", "")
    if suppressed:
        SUPPRESSED_EVENTS = {e.strip() for e in suppressed.split(",") if e.strip()}


def _event_filter(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Filter out suppressed events.

    Structlog processor signature: (logger, method_name, event_dict) -> event_dict
    """
    event_name = event_dict.get("event_name")
    if event_name and event_name in SUPPRESSED_EVENTS:
        raise structlog.DropEvent
    return event_dict


def setup_logging() -> None:
    """Configures the logging system using `structlog`.

    The function uses the TIMBAL_LOG_LEVEL and TIMBAL_LOG_FORMAT environment variables
    to determine the logging level and format. It supports both human-readable and
    JSON-based structured logging.

    Called again (every `create_app()` calls it), it reconfigures the root handler it added rather than add
    another. Calls from several threads run one at a time.
    """
    with _SETUP_LOCK:
        _setup_logging()


def _setup_logging() -> None:
    log_level = os.getenv("TIMBAL_LOG_LEVEL", "INFO")

    # Switch to human readable log output if TIMBAL_LOG_FORMAT is set to "dev"
    dev_logs = os.getenv("TIMBAL_LOG_FORMAT", "") == "dev"

    # Load suppressed events from environment
    _load_suppressed_events()

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        _event_filter,  # Filter suppressed events
    ]

    if dev_logs:
        processors.append(structlog.dev.set_exc_info)
    else:
        processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if dev_logs:
        log_renderer = structlog.dev.ConsoleRenderer(event_key="message")
    else:
        log_renderer = structlog.processors.JSONRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            log_renderer,
        ],
    )

    # Called again (a test process calls it many times), reconfigure the handler an earlier call added rather than
    # add another, which wrote every record once per call. It stays on the root logger throughout: left without a
    # handler for a moment, a module-level `logging.warning()` from another thread would call `basicConfig()`, whose
    # plain handler would stay and write every record a second time. Handlers the application added are left as
    # they are (`is True`: an object that answers any attribute, a mock, is not taken for ours).
    root = logging.getLogger()
    ours = [h for h in root.handlers if getattr(h, _HANDLER_MARK, None) is True]
    if ours:
        handler = ours[0]
        handler.acquire()  # the lock `Handler.handle()` emits under: each record goes out by one configuration
        try:
            handler.setFormatter(formatter)
            # Not `setStream()`, which flushes the stream an earlier call bound: each record was flushed as it went
            # out, and that stream may be closed by now (a test's capture).
            handler.stream = sys.stderr
        finally:
            handler.release()
        for extra in ours[1:]:  # put back next to it by code that had saved the root logger's handlers
            root.removeHandler(extra)
            extra.close()
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)
    root.setLevel(log_level)
