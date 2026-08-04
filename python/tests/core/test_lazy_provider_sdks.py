"""Provider SDK laziness: openai/anthropic must only load when actually used.

`from timbal import Agent` imports neither SDK; the first LLM call imports
only the SDK of the provider in use; error classification never imports an
SDK (an exception can only come from an already-imported one).

Each test runs in a subprocess so sys.modules assertions are not polluted by
the rest of the suite.
"""

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    """Run code in a subprocess; return the LAST stdout line (skips log lines)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


@pytest.mark.timeout(120)
class TestLazyProviderSdks:
    def test_agent_import_loads_neither_sdk(self):
        out = _run(
            "import sys\n"
            "from timbal import Agent\n"
            "print('openai' in sys.modules, 'anthropic' in sys.modules)\n"
        )
        assert out.strip() == "False False"

    def test_testmodel_agent_run_loads_neither_sdk(self):
        out = _run(
            "import asyncio, sys\n"
            "from timbal import Agent\n"
            "from timbal.core.test_model import TestModel\n"
            "async def main():\n"
            "    agent = Agent(name='x', model=TestModel(responses=['ok']), tools=[])\n"
            "    res = await agent(prompt='hi').collect()\n"
            "    assert res.status.code == 'success', res.error\n"
            "asyncio.run(main())\n"
            "print('openai' in sys.modules, 'anthropic' in sys.modules)\n"
        )
        assert out.strip() == "False False"

    def test_anthropic_client_resolution_does_not_import_openai(self):
        out = _run(
            "import sys\n"
            "from timbal.core.llm_router import _PROVIDERS, _resolve_client\n"
            "from timbal.state.context import RunContext\n"
            "ctx = RunContext(tracing_provider=None)\n"
            "client, _ = _resolve_client('anthropic', _PROVIDERS['anthropic'], 'sk-fake', None, ctx)\n"
            "assert type(client).__name__ == 'AsyncAnthropic'\n"
            "print('openai' in sys.modules, 'anthropic' in sys.modules)\n"
        )
        assert out.strip() == "False True"

    def test_openai_client_resolution_does_not_import_anthropic(self):
        out = _run(
            "import sys\n"
            "from timbal.core.llm_router import _PROVIDERS, _resolve_client\n"
            "from timbal.state.context import RunContext\n"
            "ctx = RunContext(tracing_provider=None)\n"
            "client, _ = _resolve_client('openai', _PROVIDERS['openai'], 'sk-fake', None, ctx)\n"
            "assert type(client).__name__ == 'AsyncOpenAI'\n"
            "print('openai' in sys.modules, 'anthropic' in sys.modules)\n"
        )
        assert out.strip() == "True False"

    def test_error_classification_with_single_sdk(self):
        """Classifying an anthropic error must work without importing openai."""
        out = _run(
            "import sys\n"
            "import httpx\n"
            "import anthropic\n"
            "from timbal.core.fallback_model import is_retryable_provider_error\n"
            "resp = httpx.Response(429, request=httpx.Request('POST', 'http://x'))\n"
            "exc = anthropic.RateLimitError('rate limited', response=resp, body=None)\n"
            "assert is_retryable_provider_error(exc) is True\n"
            "assert is_retryable_provider_error(ValueError('nope')) is False\n"
            "print('openai' in sys.modules)\n"
        )
        assert out.strip() == "False"

    def test_error_classification_with_no_sdk(self):
        """TestModel-only processes classify errors with empty class tuples."""
        out = _run(
            "import sys\n"
            "from timbal.core.provider_errors import provider_error_classes\n"
            "classes = provider_error_classes()\n"
            "assert all(v == () for v in classes.values()), classes\n"
            "assert not isinstance(ValueError('x'), classes['rate_limit'])\n"
            "print('openai' in sys.modules, 'anthropic' in sys.modules)\n"
        )
        assert out.strip() == "False False"
