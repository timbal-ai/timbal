"""Integration tests for the Parallel tools.

These hit the live Parallel API and require PARALLEL_API_KEY in the environment
(loaded from .env by conftest). Run with: uv run pytest -m integration python/tests/core/test_parallel.py
"""

import os

import pytest
from timbal.tools import ParallelExtract, ParallelSearch, ParallelTaskRun

pytestmark = pytest.mark.integration

_NO_KEY = not os.getenv("PARALLEL_API_KEY")
_SKIP_REASON = "PARALLEL_API_KEY not set"


@pytest.mark.skipif(_NO_KEY, reason=_SKIP_REASON)
class TestParallelSearch:
    async def test_search_returns_results(self):
        tool = ParallelSearch()
        output = await tool(
            search_queries=["Parallel.ai web search API"],
            objective="What does Parallel.ai offer?",
            max_results=3,
        ).collect()

        assert output.status.code == "success", output.error
        result = output.output
        assert "results" in result
        assert len(result["results"]) >= 1
        first = result["results"][0]
        assert first["url"]
        assert isinstance(first["excerpts"], list)

    async def test_search_respects_include_domains(self):
        tool = ParallelSearch()
        output = await tool(
            search_queries=["web search API for AI agents"],
            include_domains=["parallel.ai"],
            max_results=5,
        ).collect()

        assert output.status.code == "success", output.error
        results = output.output["results"]
        assert results, "expected at least one result"
        assert all("parallel.ai" in r["url"] for r in results)


@pytest.mark.skipif(_NO_KEY, reason=_SKIP_REASON)
class TestParallelExtract:
    async def test_extract_from_url(self):
        tool = ParallelExtract()
        output = await tool(
            urls=["https://parallel.ai/"],
            objective="What products does Parallel offer?",
        ).collect()

        assert output.status.code == "success", output.error
        results = output.output["results"]
        assert len(results) >= 1
        assert results[0]["url"]
        assert isinstance(results[0]["excerpts"], list)


@pytest.mark.skipif(_NO_KEY, reason=_SKIP_REASON)
class TestParallelTaskRun:
    async def test_task_run_text_output(self):
        tool = ParallelTaskRun()
        output = await tool(
            input="What is the capital of France? Answer in one word.",
            processor="lite",
        ).collect()

        assert output.status.code == "success", output.error
        result = output.output
        # Task result carries the produced output under "output".
        assert "output" in result
        content = result["output"].get("content")
        assert content is not None
        assert "paris" in str(content).lower()

    async def test_task_run_structured_output(self):
        tool = ParallelTaskRun()
        output = await tool(
            input="What is the capital of Japan?",
            processor="lite",
            output_schema={
                "type": "object",
                "properties": {"capital": {"type": "string"}},
                "required": ["capital"],
                "additionalProperties": False,
            },
        ).collect()

        assert output.status.code == "success", output.error
        content = output.output["output"]["content"]
        # Structured output comes back as a dict matching the schema.
        assert isinstance(content, dict)
        assert "tokyo" in str(content.get("capital", "")).lower()
