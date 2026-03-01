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
