"""Tests for Cloudflare Browser Rendering crawl tools.

Unit tests mock httpx. Live integration tests require ``CLOUDFLARE_API_TOKEN`` and
``CLOUDFLARE_ACCOUNT_ID``.

Run integration tests explicitly::

    uv run pytest python/tests/tools/test_cloudflare.py -m integration -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.core.tool import Tool
from timbal.tools.cloudflare import (
    CF_CRAWL_BILLING_NAME,
    CloudflareCrawlCancel,
    CloudflareCrawlGet,
    CloudflareCrawlStart,
)
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message

_BROWSER_SECONDS_KEY = f"{CF_CRAWL_BILLING_NAME}:browser_seconds"
_PAGES_CRAWLED_KEY = f"{CF_CRAWL_BILLING_NAME}:pages_crawled"
_REQUESTS_KEY = f"{CF_CRAWL_BILLING_NAME}:requests"


def _skip_if_cloudflare_not_configured() -> None:
    if not os.getenv("CLOUDFLARE_API_TOKEN") or not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        pytest.skip("Cloudflare integration: set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID")


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _completed_crawl_payload(
    *,
    browser_seconds: float = 134.7,
    finished: int = 50,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "success": True,
        "result": {
            "status": status,
            "browserSecondsUsed": browser_seconds,
            "total": finished,
            "finished": finished,
            "records": [],
        },
    }


async def _invoke(tool: Tool, **kwargs: Any) -> Any:
    result = await tool(**kwargs).collect()
    if result.error:
        message = result.error.get("message", result.error) if isinstance(result.error, dict) else result.error
        raise AssertionError(f"{tool.name} failed: {message}")
    return result


@pytest.mark.asyncio
async def test_crawl_get_completed_records_browser_seconds_pages_and_requests():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _completed_crawl_payload()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CloudflareCrawlGet(api_token=SecretStr("cf_test"), account_id="acct_123")
        out = await _invoke(tool, job_id="job-abc")

    assert out.usage[_BROWSER_SECONDS_KEY] == 135
    assert out.usage[_PAGES_CRAWLED_KEY] == 50
    assert out.usage[_REQUESTS_KEY] == 1
    assert "cloudflare_crawl_get:requests" not in out.usage


@pytest.mark.asyncio
async def test_crawl_get_running_poll_records_no_usage():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "success": True,
        "result": {"status": "running", "finished": 3, "total": 10, "records": []},
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CloudflareCrawlGet(api_token=SecretStr("cf_test"), account_id="acct_123")
        out = await _invoke(tool, job_id="job-abc", limit=1)

    assert _BROWSER_SECONDS_KEY not in out.usage
    assert _PAGES_CRAWLED_KEY not in out.usage
    assert _REQUESTS_KEY not in out.usage


@pytest.mark.asyncio
async def test_crawl_get_errored_records_requests_without_browser_seconds():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "success": True,
        "result": {"status": "errored", "finished": 0, "total": 5, "records": []},
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CloudflareCrawlGet(api_token=SecretStr("cf_test"), account_id="acct_123")
        out = await _invoke(tool, job_id="job-abc")

    assert _BROWSER_SECONDS_KEY not in out.usage
    assert _PAGES_CRAWLED_KEY not in out.usage
    assert out.usage[_REQUESTS_KEY] == 1


@pytest.mark.asyncio
async def test_crawl_start_does_not_record_crawl_usage():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"success": True, "result": "job-new"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CloudflareCrawlStart(api_token=SecretStr("cf_test"), account_id="acct_123")
        out = await _invoke(tool, url="https://example.com", limit=1)

    assert _BROWSER_SECONDS_KEY not in out.usage
    assert _REQUESTS_KEY not in out.usage
    assert "cloudflare_crawl_start:requests" not in out.usage


@pytest.mark.asyncio
async def test_crawl_cancel_does_not_record_crawl_usage():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b'{"success": true}'
    mock_response.json.return_value = {"success": True}

    mock_client = MagicMock()
    mock_client.delete = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CloudflareCrawlCancel(api_token=SecretStr("cf_test"), account_id="acct_123")
        out = await _invoke(tool, job_id="job-abc")

    assert _REQUESTS_KEY not in out.usage
    assert "cloudflare_crawl_cancel:requests" not in out.usage


@pytest.mark.asyncio
async def test_crawl_get_completed_uses_total_when_finished_missing():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "success": True,
        "result": {
            "status": "completed",
            "browserSecondsUsed": 10.4,
            "total": 7,
        },
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CloudflareCrawlGet(api_token=SecretStr("cf_test"), account_id="acct_123")
        out = await _invoke(tool, job_id="job-abc")

    assert out.usage[_BROWSER_SECONDS_KEY] == 10
    assert out.usage[_PAGES_CRAWLED_KEY] == 7
    assert out.usage[_REQUESTS_KEY] == 1


@pytest.mark.asyncio
async def test_crawl_get_terminal_poll_dedupes_usage_within_agent_run():
    payload = _completed_crawl_payload(browser_seconds=20.0, finished=2)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = payload

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    tool = CloudflareCrawlGet(api_token=SecretStr("cf_test"), account_id="acct_123")
    tool_name = tool.name

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        model = TestModel(
            responses=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="t1", name=tool_name, input={"job_id": "job-dedupe", "limit": 1}),
                    ],
                    stop_reason="tool_use",
                ),
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="t2", name=tool_name, input={"job_id": "job-dedupe"})],
                    stop_reason="tool_use",
                ),
                Message(role="assistant", content=[TextContent(text="done")], stop_reason="end_turn"),
            ]
        )
        agent = Agent(name="cf_agent", model=model, tools=[tool])
        out = await agent(prompt="Poll the crawl job twice.").collect()

    assert out.usage.get(_REQUESTS_KEY) == 1
    assert out.usage.get(_BROWSER_SECONDS_KEY) == 20


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crawl_live_start_poll_records_usage():
    """Start a 1-page crawl, poll until terminal, assert usage keys."""
    _skip_if_cloudflare_not_configured()

    start_tool = CloudflareCrawlStart()
    start_out = await _invoke(start_tool, url="https://example.com", limit=1, render=True)
    job_id = start_out.output["job_id"]

    get_tool = CloudflareCrawlGet()
    terminal = None
    for _ in range(60):
        poll = await get_tool(job_id=job_id, limit=1).collect()
        status = poll.output.get("result", {}).get("status")
        if status in {
            "completed",
            "errored",
            "cancelled_due_to_timeout",
            "cancelled_due_to_limits",
            "cancelled_by_user",
        }:
            terminal = poll
            break
        import asyncio

        await asyncio.sleep(2)

    assert terminal is not None, "crawl did not reach a terminal status within timeout"
    assert terminal.status.code == "success"
    assert terminal.usage.get(_REQUESTS_KEY) == 1
    if terminal.output.get("result", {}).get("browserSecondsUsed"):
        assert _BROWSER_SECONDS_KEY in terminal.usage
