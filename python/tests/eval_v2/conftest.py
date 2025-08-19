"""
Shared test fixtures for eval_v2 tests.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from timbal.server.utils import ModuleSpec, load_module

load_dotenv()


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_llm: marks tests as requiring a real LLM API key"
    )


@pytest.fixture(scope="session", autouse=True)
def configure_test_llm():
    """Configure LLM for tests - loads environment configuration like core_v2."""
    # The load_dotenv() call at module level will load .env file
    # Just ensure we have the API key available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment. Please set it in .env file or environment.")
    return api_key


@pytest.fixture
def fixtures_dir():
    """Get the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_agent(fixtures_dir):
    """Load the sample agent from fixtures."""
    agent_file = fixtures_dir / "sample_agent.py"
    module_spec = ModuleSpec(
        path=agent_file,
        object_name="agent"
    )
    return load_module(module_spec)


@pytest.fixture
def simple_test_file(fixtures_dir):
    """Get the simple test file path."""
    return fixtures_dir / "eval_simple_test.yaml"


@pytest.fixture
def agent_test_file(fixtures_dir):
    """Get the agent test file path."""
    return fixtures_dir / "eval_agent.yaml"
