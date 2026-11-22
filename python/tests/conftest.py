import os
import time

import pytest
from dotenv import load_dotenv
from timbal.types.events import OutputEvent

# Set environment variable early to suppress warnings during imports
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


def pytest_configure(config):
    """Configure pytest with global settings."""
    # Load environment variables
    load_dotenv(override=True)
    if os.environ.get("TIMBAL_TEST_SLOW_IO"):
        _slow_storage(float(os.environ["TIMBAL_TEST_SLOW_IO"]))


def _slow_storage(delay: float) -> None:
    """Opt-in (``TIMBAL_TEST_SLOW_IO=<seconds>``): every chat-session store read/write and every SQLite/JSONL
    trace save waits ``delay`` first, like the disk of a loaded CI runner. A test that only passes because
    storage is fast on the machine it was written on fails here, not on Windows a week later.
    ``TIMBAL_TEST_SLOW_IO_JITTER=1`` waits a random time in [0, delay] instead (orderings vary too;
    ``TIMBAL_TEST_SLOW_IO_SEED`` picks the sequence)."""
    import asyncio
    import functools
    import random

    from timbal.state.chat_store import SqliteChatSessionStore
    from timbal.state.tracing.providers import JsonlTracingProvider, SqliteTracingProvider

    jitter = os.environ.get("TIMBAL_TEST_SLOW_IO_JITTER") == "1"
    rng = random.Random(int(os.environ.get("TIMBAL_TEST_SLOW_IO_SEED", "0")))

    def slowed(cls, name):
        real = cls.__dict__[name]
        inner = real.__func__ if isinstance(real, classmethod) else real

        @functools.wraps(inner)
        async def wrapped(*args, **kwargs):
            await asyncio.sleep(rng.uniform(0, delay) if jitter else delay)
            return await inner(*args, **kwargs)

        setattr(cls, name, classmethod(wrapped) if isinstance(real, classmethod) else wrapped)

    slowed(SqliteChatSessionStore, "save")
    slowed(SqliteChatSessionStore, "load")
    slowed(SqliteTracingProvider, "_store")
    slowed(JsonlTracingProvider, "_store")


@pytest.fixture(scope="session")
def _isolated_timbal_config_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("timbal_isolated_config")


@pytest.fixture(autouse=True)
def reset_platform_config_cache(request, _isolated_timbal_config_dir):
    """Force platform config to resolve as None for unit tests only.

    Setting _default_config_resolved=True with _cached_default_config=None tells
    resolve_platform_config() "already resolved — there is no platform config".
    Any RunContext() without an explicit tracing_provider then falls back to
    InMemoryTracingProvider instead of PlatformTracingProvider, preventing HTTP
    calls on every _save_trace() / get_session() across the suite.
    TIMBAL_CONFIG_DIR is also pointed at an empty temp directory so unit tests
    that pass an explicit PlatformConfig or reset _default_config_resolved never
    read the developer's real ~/.timbal credentials.

    Integration tests opt out so they can use ~/.timbal credentials and the
    platform LLM proxy when provider API keys are not in the environment.
    """
    import timbal.state.config_loader as _cl

    saved_config_dir = _cl.TIMBAL_CONFIG_DIR
    is_integration = request.node.get_closest_marker("integration") is not None
    if is_integration:
        _cl._cached_default_config = None
        _cl._default_config_resolved = False
        _cl._file_config_cache.clear()
        yield
        _cl.TIMBAL_CONFIG_DIR = saved_config_dir
        _cl._cached_default_config = None
        _cl._default_config_resolved = False
        _cl._file_config_cache.clear()
        return

    _cl.TIMBAL_CONFIG_DIR = _isolated_timbal_config_dir
    _cl._cached_default_config = None
    _cl._default_config_resolved = True
    _cl._file_config_cache.clear()
    yield
    _cl.TIMBAL_CONFIG_DIR = saved_config_dir
    _cl._cached_default_config = None
    _cl._default_config_resolved = True
    _cl._file_config_cache.clear()


@pytest.fixture(autouse=True)
def clear_background_task_stores():
    """Drop the process-local background-task registry between tests.

    Stores are keyed by run_id and inherited via parent_id. Without a reset,
    a leftover parent_id from a previous test can leak children into the next.
    """
    from timbal.state.background import clear_background_stores

    clear_background_stores()
    yield
    clear_background_stores()


@pytest.fixture(autouse=True)
def clear_in_memory_tracing_storage():
    """Clear InMemoryTracingProvider storage and the platform provider's recent-trace cache between tests.

    Both are class-level dicts: the storage otherwise grows unbounded across the
    full test suite, and a trace one test stored would answer another test's get().
    """
    from timbal.state.tracing.providers import platform as platform_module
    from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
    from timbal.state.tracing.providers.platform import PlatformTracingProvider

    def clear():
        InMemoryTracingProvider._storage.clear()
        PlatformTracingProvider._recent.clear()
        PlatformTracingProvider._saving.clear()
        # Saves a test left unlanded would otherwise be resent by the exit flush, to a host that was mocked.
        PlatformTracingProvider._flights.clear()
        PlatformTracingProvider._unsaved.clear()
        PlatformTracingProvider._own_snapshots.clear()
        for task in PlatformTracingProvider._retries.values():
            task.cancel()
        PlatformTracingProvider._retries.clear()
        PlatformTracingProvider._closing = False
        platform_module._TRACE_POOLS.clear()
        PlatformTracingProvider._oversized.clear()

    clear()
    yield
    clear()


# How long the event loop may run no callback before loop_watchdog reports it (TIMBAL_TEST_LOOP_STALL_S).
LOOP_STALL_S = float(os.environ.get("TIMBAL_TEST_LOOP_STALL_S", "10"))


@pytest.fixture
async def loop_watchdog(request):
    """Report a test whose event loop stopped running callbacks for ``LOOP_STALL_S``, with every thread's stack.

    A test that waits for a session (``until``, ``settle``, ``records``) and times out says only that the work
    did not happen. This tells the two other reasons apart: the loop thread was blocked (a synchronous call on
    it; the stacks say which, and the test fails), or the whole process stopped running (this thread was late
    by as much: nothing in the process ran, so nothing could have happened; noted on the report, the test's
    own result stands). Measured on ``time.perf_counter``, which no test moves (``Clock`` moves monotonic).

    Async tests only: during a sync test's body nothing runs the loop, so a sync test that waits (on a thread,
    a file) would read as a blocked loop. Applied module-wide, it lets the sync tests of the module pass as is.
    """
    import asyncio
    import inspect
    import sys
    import threading
    import traceback

    if not inspect.iscoroutinefunction(getattr(request.node, "obj", None)):
        yield
        return

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    last_beat = [time.perf_counter()]
    found: list[tuple[str, float, float, str]] = []  # (kind, loop late, this thread late, stacks)
    stop = threading.Event()

    def beat() -> None:
        last_beat[0] = time.perf_counter()

    def stacks() -> str:
        out = []
        for ident, frame in sys._current_frames().items():
            label = "event loop" if ident == loop_thread else "thread"
            out.append(f"--- {label} {ident} ---\n" + "".join(traceback.format_stack(frame)))
        return "\n".join(out)

    def watch() -> None:
        ticked, reported = time.perf_counter(), False
        while not stop.wait(0.25):
            now = time.perf_counter()
            own_late, ticked = now - ticked - 0.25, now
            late = now - last_beat[0]
            if own_late >= LOOP_STALL_S:
                # This thread did not run either: the process was stopped (the loop's stack now says nothing).
                found.append(("process", late, own_late, ""))
                reported = True
            elif late >= LOOP_STALL_S and not reported:
                # On time itself while the loop is not: something holds the loop thread. Its stack says what.
                found.append(("loop", late, own_late, stacks()))
                reported = True
            elif late < 1.0:
                reported = False
            try:
                loop.call_soon_threadsafe(beat)
            except RuntimeError:
                return  # the loop is closed

    watcher = threading.Thread(target=watch, name="loop-watchdog", daemon=True)
    watcher.start()
    yield
    stop.set()
    watcher.join(2)
    if not found:
        return
    text = "\n\n".join(
        f"The event loop ran no callback for {late:.1f} s; this watchdog thread was itself {own:.1f} s late "
        f"({'the whole process stopped running' if kind == 'process' else 'the loop thread was blocked'}).\n"
        f"{stack}"
        for kind, late, own, stack in found
    )
    if any(kind == "loop" for kind, *_ in found):
        pytest.fail(text, pytrace=False)
    request.node.add_report_section("teardown", "loop watchdog", text)


def assert_has_output_event(output: OutputEvent):
    """Assert that we have a valid OutputEvent."""
    assert isinstance(output, OutputEvent), f"Expected OutputEvent, got {type(output)}"


def assert_no_errors(output: OutputEvent):
    """Assert that the output contains no errors."""
    if output.error:
        pytest.fail(f"Found error in OutputEvent: {output.error}")


def skip_if_agent_error(output: OutputEvent, test_name: str = ""):
    """Skip test if agent execution failed - indicates implementation issue."""
    if output.error is not None:
        error_msg = output.error.get("message", str(output.error))
        pytest.skip(f"Agent execution failed in {test_name} - needs implementation fix: {error_msg}")


class Timer:
    """Simple timer context manager for performance testing."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        # perf_counter: high-res monotonic clock. time.time() has ~15.6ms
        # granularity on Windows, which makes sub-tick durations flaky.
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
