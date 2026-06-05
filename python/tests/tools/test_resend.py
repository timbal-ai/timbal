"""Tests for Resend tools.

Unit tests use mocked httpx. Live integration tests require ``RESEND_API_KEY`` with
full access (send-only keys fail list/get/update/cancel with 401).

Run integration tests explicitly::

    uv run python -m pytest python/tests/tools/test_resend.py -m integration -v
"""

from __future__ import annotations

import base64
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.core.tool import Tool
from timbal.tools.resend import (
    ResendCancelEmail,
    ResendGetEmail,
    ResendGetEmailAttachment,
    ResendGetReceivedAttachment,
    ResendGetReceivedEmail,
    ResendListEmailAttachments,
    ResendListEmails,
    ResendListReceivedAttachments,
    ResendListReceivedEmails,
    ResendSendBatch,
    ResendSendEmail,
    ResendUpdateEmail,
)

FROM = os.getenv("RESEND_FROM_EMAIL") or "onboarding@resend.dev"
TO = "delivered@resend.dev"


def _skip_if_resend_not_configured() -> None:
    if not os.getenv("RESEND_API_KEY"):
        pytest.skip("Resend integration: set RESEND_API_KEY in the environment or .env")


async def _invoke(tool: Tool, **kwargs: Any) -> Any:
    result = await tool(**kwargs).collect()
    if result.error:
        message = result.error.get("message", result.error) if isinstance(result.error, dict) else result.error
        raise AssertionError(f"{tool.name} failed: {message}")
    return result.output


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_resend_send_email_posts_payload_and_user_agent():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"id": "email-123"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = ResendSendEmail(api_key=SecretStr("re_test_key"))
        out = await tool.handler(
            to=["delivered@resend.dev"],
            subject="hello world",
            from_="Acme <onboarding@resend.dev>",
            html="<p>it works!</p>",
            text=None,
            cc=None,
            bcc=None,
            reply_to=None,
            scheduled_at=None,
            headers=None,
            attachments=None,
            tags=None,
            topic_id=None,
            template=None,
            idempotency_key=None,
        )

    assert out == {"id": "email-123"}
    mock_client.post.assert_awaited_once()
    url = mock_client.post.await_args[0][0]
    assert url == "https://api.resend.com/emails"
    headers = mock_client.post.await_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer re_test_key"
    assert headers["User-Agent"] == "timbal-resend-tools/1.0"
    body = mock_client.post.await_args.kwargs["json"]
    assert body["from"] == "Acme <onboarding@resend.dev>"
    assert body["to"] == ["delivered@resend.dev"]
    assert body["html"] == "<p>it works!</p>"


@pytest.mark.asyncio
async def test_resend_send_email_uses_default_from():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"id": "email-456"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = ResendSendEmail(
            api_key=SecretStr("re_test_key"),
            default_from="Default <default@example.com>",
        )
        await tool.handler(
            to=["user@example.com"],
            subject="test",
            from_=None,
            html="<p>hi</p>",
            text=None,
            cc=None,
            bcc=None,
            reply_to=None,
            scheduled_at=None,
            headers=None,
            attachments=None,
            tags=None,
            topic_id=None,
            template=None,
            idempotency_key=None,
        )

    body = mock_client.post.await_args.kwargs["json"]
    assert body["from"] == "Default <default@example.com>"


@pytest.mark.asyncio
async def test_resend_list_received_emails_gets_with_after():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"object": "list", "has_more": False, "data": []}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = ResendListReceivedEmails(api_key=SecretStr("re_test_key"))
        out = await tool.handler(after="prev-email-id")

    assert out == {"object": "list", "has_more": False, "data": []}
    mock_client.get.assert_awaited_once()
    url = mock_client.get.await_args[0][0]
    assert url == "https://api.resend.com/emails/receiving"
    params = mock_client.get.await_args.kwargs["params"]
    assert params == {"after": "prev-email-id"}
    headers = mock_client.get.await_args.kwargs["headers"]
    assert headers["User-Agent"] == "timbal-resend-tools/1.0"


@pytest.mark.asyncio
async def test_resend_cancel_email_posts_cancel_path():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"object": "email", "id": "49a3999c-0ce1-4ea6-ab68-afcd6dc2e794"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = ResendCancelEmail(api_key=SecretStr("re_test_key"))
        out = await tool.handler(email_id="49a3999c-0ce1-4ea6-ab68-afcd6dc2e794")

    assert out["id"] == "49a3999c-0ce1-4ea6-ab68-afcd6dc2e794"
    mock_client.post.assert_awaited_once()
    url = mock_client.post.await_args[0][0]
    assert url == "https://api.resend.com/emails/49a3999c-0ce1-4ea6-ab68-afcd6dc2e794/cancel"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resend_live_all_tools() -> None:
    """End-to-end smoke test for all 12 Resend email + inbound tools."""
    _skip_if_resend_not_configured()

    sent_id: str | None = None
    attachment_email_id: str | None = None
    scheduled_id: str | None = None
    scheduled_cancel_id: str | None = None

    out = await _invoke(
        ResendSendEmail(),
        from_=f"Timbal Test <{FROM}>",
        to=[TO],
        subject="pytest resend send",
        html="<p>ResendSendEmail OK</p>",
    )
    sent_id = out["id"]

    batch = await _invoke(
        ResendSendBatch(),
        emails=[
            {"from": f"Timbal Test <{FROM}>", "to": [TO], "subject": "pytest batch 1", "html": "<p>1</p>"},
            {"from": f"Timbal Test <{FROM}>", "to": [TO], "subject": "pytest batch 2", "html": "<p>2</p>"},
        ],
    )
    assert len(batch["data"]) == 2

    attachment_out = await _invoke(
        ResendSendEmail(),
        from_=f"Timbal Test <{FROM}>",
        to=[TO],
        subject="pytest attachment",
        html="<p>attachment</p>",
        attachments=[{"filename": "test.txt", "content": base64.b64encode(b"hello").decode()}],
    )
    attachment_email_id = attachment_out["id"]

    future = (datetime.now(UTC) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    future_cancel = (datetime.now(UTC) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    scheduled = await _invoke(
        ResendSendEmail(),
        from_=f"Timbal Test <{FROM}>",
        to=[TO],
        subject="pytest scheduled update",
        html="<p>update</p>",
        scheduled_at=future,
    )
    scheduled_id = scheduled["id"]

    scheduled_cancel = await _invoke(
        ResendSendEmail(),
        from_=f"Timbal Test <{FROM}>",
        to=[TO],
        subject="pytest scheduled cancel",
        html="<p>cancel</p>",
        scheduled_at=future_cancel,
    )
    scheduled_cancel_id = scheduled_cancel["id"]

    listed = await _invoke(ResendListEmails())
    assert "data" in listed

    got = await _invoke(ResendGetEmail(), email_id=sent_id)
    assert got["id"] == sent_id

    updated = await _invoke(
        ResendUpdateEmail(),
        email_id=scheduled_id,
        scheduled_at=(datetime.now(UTC) + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    assert updated["id"] == scheduled_id

    cancelled = await _invoke(ResendCancelEmail(), email_id=scheduled_cancel_id)
    assert cancelled["id"] == scheduled_cancel_id

    attachments = await _invoke(ResendListEmailAttachments(), email_id=attachment_email_id)
    assert "data" in attachments
    assert len(attachments["data"]) >= 1
    attachment_id = attachments["data"][0]["id"]

    attachment = await _invoke(
        ResendGetEmailAttachment(),
        email_id=attachment_email_id,
        attachment_id=attachment_id,
    )
    assert attachment["id"] == attachment_id

    received_list = await _invoke(ResendListReceivedEmails())
    assert "data" in received_list

    if received_list["data"]:
        received_id = received_list["data"][0]["id"]
        received = await _invoke(ResendGetReceivedEmail(), email_id=received_id)
        assert received["id"] == received_id

        received_attachments = await _invoke(ResendListReceivedAttachments(), email_id=received_id)
        assert "data" in received_attachments
        if received_attachments["data"]:
            received_attachment_id = received_attachments["data"][0]["id"]
            received_attachment = await _invoke(
                ResendGetReceivedAttachment(),
                email_id=received_id,
                attachment_id=received_attachment_id,
            )
            assert received_attachment["id"] == received_attachment_id
