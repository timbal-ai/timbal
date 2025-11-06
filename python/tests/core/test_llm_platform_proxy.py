"""Tests for LLM platform proxy configuration.

IMPORTANT: These tests require a .env.test_llm_platform_proxy file in the tests/core/ directory.
Copy .env.test_llm_platform_proxy.example to .env.test_llm_platform_proxy and configure the following variables:
    - TIMBAL_API_HOST (e.g., api.timbal.ai)
    - TIMBAL_ORG_ID (your organization ID)
    - TIMBAL_PROJECT_ID (your project ID)

All tests in this file will be skipped if .env.test_llm_platform_proxy is not found.
"""
import os
from pathlib import Path

import pytest

# NOTE: Timbal imports are delayed until after environment setup in each test
# This allows environment variables to be set before the libraries are loaded

# Check for .env.test_llm_platform_proxy file
TEST_ENV_FILE = Path(__file__).parent / ".env.test_llm_platform_proxy"
SKIP_REASON = (
    "Platform proxy tests require .env.test_llm_platform_proxy file. "
    "Copy .env.test_llm_platform_proxy.example to .env.test_llm_platform_proxy and configure your platform credentials."
)


def load_test_env():
    """Load environment variables from .env.test_llm_platform_proxy file."""
    if not TEST_ENV_FILE.exists():
        return False
    
    with open(TEST_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()
    return True


@pytest.mark.skipif(not TEST_ENV_FILE.exists(), reason=SKIP_REASON)
class TestPlatformProxy:
    """Test that agents use platform proxy when configured."""

    def setup_method(self):
        """Set up platform config via environment variables."""
        # Save the entire environment
        self.saved_environ = os.environ.copy()
        
        # Clear the entire environment
        os.environ.clear()
        
        # Load only the test environment variables from .env.test_llm_platform_proxy
        load_test_env()

    def teardown_method(self):
        """Clean up."""
        # Restore the entire original environment
        os.environ.clear()
        os.environ.update(self.saved_environ)

    @pytest.mark.asyncio
    async def test_openai_uses_proxy(self):
        """Test OpenAI uses platform proxy."""
        from timbal import Agent
        from timbal.state import RunContext, set_run_context
        
        run_context = RunContext()
        set_run_context(run_context)
        agent = Agent(name="test", model="openai/gpt-4o-mini")
        await agent(prompt="hello").collect()

    @pytest.mark.asyncio
    async def test_anthropic_uses_proxy(self):
        """Test Anthropic uses platform proxy."""
        from timbal import Agent
        from timbal.state import RunContext, set_run_context
        
        run_context = RunContext()
        set_run_context(run_context)
        agent = Agent(name="test", model="anthropic/claude-haiku-4-5", model_params={"max_tokens": 1024})
        await agent(prompt="hello").collect()

    @pytest.mark.asyncio
    async def test_gemini_uses_proxy(self):
        """Test Google Gemini uses platform proxy."""
        from timbal import Agent
        from timbal.state import RunContext, set_run_context
        
        run_context = RunContext()
        set_run_context(run_context)
        agent = Agent(name="test", model="google/gemini-2.5-flash-lite")
        await agent(prompt="hello").collect()

    @pytest.mark.asyncio
    async def test_togetherai_uses_proxy(self):
        """Test TogetherAI uses platform proxy."""
        from timbal import Agent
        from timbal.state import RunContext, set_run_context
        
        run_context = RunContext()
        set_run_context(run_context)
        agent = Agent(name="test", model="togetherai/deepseek-ai/DeepSeek-V3.1")
        await agent(prompt="hello").collect()
