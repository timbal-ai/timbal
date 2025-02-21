import logging
import os

import structlog


def setup_logging() -> None:
    """Configures the logging system using `structlog`.

    The function uses the TIMBAL_LOG_LEVEL and TIMBAL_LOG_FORMAT environment variables 
    to determine the logging level and format. It supports both human-readable and 
    JSON-based structured logging.
    """
    log_level = os.getenv("TIMBAL_LOG_LEVEL", "INFO")

    # Switch to human readable log output if TIMBAL_LOG_FORMAT is set to "dev"
    dev_logs = os.getenv("TIMBAL_LOG_FORMAT", "") == "dev"

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if dev_logs:
        processors.append(structlog.dev.set_exc_info)
    else:
        processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
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

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(log_level)
