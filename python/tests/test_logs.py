"""`timbal.logs.setup_logging()` called more than once (every `create_app()` does) keeps one handler of its own."""

import io
import json
import logging
import sys
from unittest import mock

import pytest
import structlog
from timbal.logs import setup_logging


@pytest.fixture
def root_logging(monkeypatch):
    """The root logger and structlog's config as they were, put back after the test."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    configured, config = structlog.is_configured(), structlog.get_config()
    monkeypatch.delenv("TIMBAL_SUPPRESS_EVENTS", raising=False)
    monkeypatch.delenv("TIMBAL_LOG_FORMAT", raising=False)
    monkeypatch.delenv("TIMBAL_LOG_LEVEL", raising=False)
    try:
        yield root, handlers
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(level)
        if configured:
            structlog.configure(**config)
        else:
            structlog.reset_defaults()


def _added(root, before):
    return [h for h in root.handlers if h not in before]


class TestSetupLoggingTwice:
    def test_one_handler_and_one_line_per_record(self, root_logging, monkeypatch):
        root, before = root_logging
        out = io.StringIO()
        monkeypatch.setattr(sys, "stderr", out)  # StreamHandler() binds sys.stderr as it is made

        setup_logging()
        setup_logging()

        assert len(_added(root, before)) == 1
        logging.getLogger("timbal.tests.logs").warning("stdlib record")
        structlog.get_logger("timbal.tests.logs").warning("structlog record")
        lines = out.getvalue().splitlines()
        assert [json.loads(line)["event"] for line in lines] == ["stdlib record", "structlog record"]

    def test_a_later_call_takes_its_settings(self, root_logging, monkeypatch):
        root, before = root_logging
        first, second = io.StringIO(), io.StringIO()
        monkeypatch.setenv("TIMBAL_LOG_FORMAT", "dev")
        monkeypatch.setattr(sys, "stderr", first)
        setup_logging()

        monkeypatch.setenv("TIMBAL_LOG_FORMAT", "json")
        monkeypatch.setenv("TIMBAL_LOG_LEVEL", "WARNING")
        monkeypatch.setattr(sys, "stderr", second)
        setup_logging()

        assert len(_added(root, before)) == 1
        logging.getLogger("timbal.tests.logs").info("below the level")
        logging.getLogger("timbal.tests.logs").warning("at the level")
        assert first.getvalue() == ""
        assert [json.loads(line)["event"] for line in second.getvalue().splitlines()] == ["at the level"]

    def test_leaves_the_applications_handlers(self, root_logging, monkeypatch):
        root, before = root_logging
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        theirs = io.StringIO()
        app_handler = logging.StreamHandler(theirs)
        root.addHandler(app_handler)

        setup_logging()
        setup_logging()

        assert app_handler in root.handlers
        assert len(_added(root, before)) == 2  # the application's, and one of timbal's
        logging.getLogger("timbal.tests.logs").warning("to both")
        assert theirs.getvalue().splitlines() == ["to both"]

    def test_leaves_a_handler_that_answers_any_attribute(self, root_logging, monkeypatch):
        """A mock has every attribute, the mark included: it is still not one of timbal's."""
        root, before = root_logging
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        theirs = mock.MagicMock()
        root.addHandler(theirs)

        setup_logging()
        setup_logging()

        assert theirs in root.handlers
        assert len(_added(root, before)) == 2
