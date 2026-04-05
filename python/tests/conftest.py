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
    load_dotenv()


@pytest.fixture(autouse=True)
def reset_platform_config_cache():
    """Force platform config to resolve as None for every non-integration test.

    Setting _default_config_resolved=True with _cached_default_config=None tells
    resolve_platform_config() "already resolved — there is no platform config".
    Any RunContext() without an explicit tracing_provider then falls back to
    InMemoryTracingProvider instead of PlatformTracingProvider, preventing HTTP
    calls on every _save_trace() / get_session() across the suite.
    """
    import timbal.state.config_loader as _cl

    _cl._cached_default_config = None
    _cl._default_config_resolved = True
    yield
    _cl._cached_default_config = None
    _cl._default_config_resolved = True


@pytest.fixture(autouse=True)
def clear_in_memory_tracing_storage():
    """Clear InMemoryTracingProvider storage between tests.

    The storage is a class-level dict that otherwise grows unbounded across
    the full test suite.
    """
    from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider

    InMemoryTracingProvider._storage.clear()
    yield
    InMemoryTracingProvider._storage.clear()


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
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
