"""`timbal.logs.setup_logging()` called more than once (every `create_app()` does) keeps one handler of its own."""

import contextlib
import io
import json
import logging
import sys
from unittest import mock

import pytest
import structlog
from timbal import logs
from timbal.logs import _HANDLER_MARK, setup_logging


def _ours(root):
    return [h for h in root.handlers if getattr(h, _HANDLER_MARK, None) is True]


@contextlib.contextmanager
def _isolated_root_logging():
    """The root logger with no handler of timbal's; on the way out, the root logger, the logger class, structlog's
    config and the `TIMBAL_*` logging variables as they were.

    A handler an earlier `create_app()` of this process left is taken off and put back as it was: `setup_logging()`
    would reconfigure it, pointing it at the test's stream. The root logger's handlers come back in their order, a
    handler's filter may change the record the next one formats. pytest adds and removes the same handler objects
    in a test's setup, call and teardown, so the list taken at setup is the one to put back at teardown.
    """
    root = logging.getLogger()
    original = root.handlers[:]
    for handler in _ours(root):
        root.removeHandler(handler)
    handlers, level, logger_class = root.handlers[:], root.level, logging.getLoggerClass()
    configured, config = structlog.is_configured(), structlog.get_config()
    with pytest.MonkeyPatch.context() as env:
        for name in ("TIMBAL_SUPPRESS_EVENTS", "TIMBAL_LOG_FORMAT", "TIMBAL_LOG_LEVEL"):
            env.delenv(name, raising=False)
        try:
            yield root, handlers
        finally:
            added = [h for h in root.handlers if h not in original]
            root.handlers[:] = original  # one step: never without a handler, nor with the test's and these at once
            for handler in added:
                handler.close()
            root.setLevel(level)
            logging.setLoggerClass(logger_class)  # `structlog.stdlib.LoggerFactory()` sets the process's logger class
            if configured:
                structlog.configure(**config)
            else:
                structlog.reset_defaults()


@pytest.fixture
def root_logging():
    with _isolated_root_logging() as isolated:
        yield isolated


def _added(root, before):
    return [h for h in root.handlers if h not in before]


def _events(stream):
    return [json.loads(line)["event"] for line in stream.getvalue().splitlines()]


def _then_logs(method):
    """`method`, then a module-level `logging.warning()`, as another thread's could come right after it."""

    def wrapper(handler):
        method(handler)
        logging.warning("meanwhile")

    return wrapper


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
        assert _events(out) == ["stdlib record", "structlog record"]

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
        assert _events(second) == ["at the level"]

    def test_the_root_logger_keeps_a_handler_while_reconfigured(self, root_logging, monkeypatch):
        """A module-level `logging.warning()` that finds the root logger without a handler calls `basicConfig()`,
        whose plain handler stays and writes every record a second time."""
        root, before = root_logging
        out = io.StringIO()
        monkeypatch.setattr(sys, "stderr", out)
        pytests = root.handlers[:]  # pytest's: with them on it, the root logger is never without a handler
        for handler in pytests:
            root.removeHandler(handler)
        try:
            setup_logging()
            with (
                mock.patch.object(root, "addHandler", _then_logs(root.addHandler)),
                mock.patch.object(root, "removeHandler", _then_logs(root.removeHandler)),
            ):
                setup_logging()

            assert [h for h in root.handlers if h not in _ours(root)] == []
            out.seek(0)
            out.truncate()
            logging.getLogger("timbal.tests.logs").warning("after")
            assert _events(out) == ["after"]
        finally:
            for handler in pytests:
                root.addHandler(handler)

    @pytest.mark.usefixtures("root_logging")
    def test_a_stream_an_earlier_call_bound_may_be_closed(self, monkeypatch, tmp_path):
        """A test's captured stderr is closed once the test is over, and the next `create_app()` must not fail."""
        # A file, not a `StringIO`: a closed `StringIO` takes `flush()`, a closed file raises, as pytest's capture does.
        first, second = open(tmp_path / "first.log", "w"), io.StringIO()  # noqa: SIM115
        monkeypatch.setattr(sys, "stderr", first)
        setup_logging()
        first.close()

        monkeypatch.setattr(sys, "stderr", second)
        setup_logging()

        logging.getLogger("timbal.tests.logs").warning("to the open one")
        assert _events(second) == ["to the open one"]

    def test_two_of_its_handlers_become_one(self, root_logging, monkeypatch):
        """Code that saved the root logger's handlers can put an earlier one of timbal's back next to the current
        one."""
        root, before = root_logging
        out = io.StringIO()
        monkeypatch.setattr(sys, "stderr", out)
        setup_logging()
        put_back = logging.StreamHandler(out)
        setattr(put_back, _HANDLER_MARK, True)
        root.addHandler(put_back)

        setup_logging()

        assert len(_added(root, before)) == 1
        logging.getLogger("timbal.tests.logs").warning("once")
        assert _events(out) == ["once"]

    @pytest.mark.usefixtures("root_logging")
    def test_a_call_holds_the_setup_lock(self, monkeypatch):
        """Two calls at once, on two threads, could each find no handler of timbal's and both add one."""
        held = []
        configure = structlog.configure
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        monkeypatch.setattr(
            structlog, "configure", lambda **kw: (held.append(logs._SETUP_LOCK.locked()), configure(**kw))
        )

        setup_logging()

        assert held == [True]

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


class TestIsolatedRootLogging:
    """The isolation the tests above run in puts back what `setup_logging()` changes in the process."""

    @pytest.fixture
    def earlier_then_theirs(self):
        """A handler of timbal's an earlier `create_app()` left, then one of the application's, after it."""
        root = logging.getLogger()
        earlier, theirs = logging.StreamHandler(io.StringIO()), logging.StreamHandler(io.StringIO())
        setattr(earlier, _HANDLER_MARK, True)
        root.addHandler(earlier)
        root.addHandler(theirs)
        try:
            yield earlier, theirs
        finally:
            root.removeHandler(earlier)
            root.removeHandler(theirs)

    def test_takes_off_an_earlier_handler_of_timbals_and_puts_every_handler_back_in_order(self, earlier_then_theirs):
        earlier, theirs = earlier_then_theirs
        root = logging.getLogger()
        expected = root.handlers[:]

        with _isolated_root_logging(), pytest.MonkeyPatch.context() as mp:
            assert earlier not in root.handlers
            assert theirs in root.handlers
            mp.setattr(sys, "stderr", io.StringIO())
            setup_logging()  # adds the test's own handler of timbal's, taken off on the way out

        assert root.handlers == expected

    def test_puts_the_logger_class_back(self):
        process_class = logging.getLoggerClass()  # an earlier `create_app()` of this process may have set structlog's
        logging.setLoggerClass(logging.Logger)
        try:
            with _isolated_root_logging(), pytest.MonkeyPatch.context() as mp:
                mp.setattr(sys, "stderr", io.StringIO())
                setup_logging()
                assert logging.getLoggerClass() is not logging.Logger

            assert logging.getLoggerClass() is logging.Logger
        finally:
            logging.setLoggerClass(process_class)
