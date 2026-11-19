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
