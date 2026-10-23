"""Unit tests for gmail_send's optional extra headers (httpx and the integration mocked)."""

import base64
from email import policy
from email.parser import Parser
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from timbal.platform.integrations import Integration
from timbal.tools.gmail import GmailSend


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = payload
    return response


def _tool() -> GmailSend:
    tool = GmailSend()
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"token": "token"})
    tool.integration = integration
    return tool


async def _send(**overrides) -> tuple[str, MagicMock]:
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_response({"id": "msg1"}))
    args = {
        "to": "guest@example.com",
        "subject": "Timbal AI <> Palex · mañana 09:00",
        "body": "Hola",
        "cc": None,
        "bcc": None,
        "attachment_urls": None,
        "message_format": "text",
        "email_signature": None,
        **overrides,
    }
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        await _tool().handler(**args)
    raw = mock_client.post.await_args.kwargs["json"]["raw"]
    return base64.urlsafe_b64decode(raw.encode()).decode(), mock_client


def _header_block(message: str) -> list[str]:
    return message.split("\r\n\r\n", 1)[0].split("\r\n")


@pytest.mark.asyncio
async def test_list_unsubscribe_headers_land_in_the_message_headers():
    message, _ = await _send(
        headers={
            "List-Unsubscribe": "<https://closer.example/m/u/abc>, <mailto:unsubscribe@example.com>",
            "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
        }
    )
    block = _header_block(message)
    assert "List-Unsubscribe: <https://closer.example/m/u/abc>, <mailto:unsubscribe@example.com>" in block
    assert "List-Unsubscribe-Post: List-Unsubscribe=One-Click" in block
    # After the subject, before the body.
    assert block.index("List-Unsubscribe-Post: List-Unsubscribe=One-Click") > next(
        i for i, line in enumerate(block) if line.startswith("Subject:")
    )


@pytest.mark.asyncio
async def test_header_names_are_matched_case_insensitively_and_written_canonically():
    message, _ = await _send(headers={"list-unsubscribe-post": "List-Unsubscribe=One-Click"})
    assert "List-Unsubscribe-Post: List-Unsubscribe=One-Click" in _header_block(message)


@pytest.mark.asyncio
async def test_headers_also_go_on_multipart_messages():
    message, _ = await _send(
        message_format="html",
        body="<p>Hola</p>",
        headers={"List-Unsubscribe": "<https://closer.example/m/u/abc>"},
    )
    block = _header_block(message)
    assert "List-Unsubscribe: <https://closer.example/m/u/abc>" in block
    assert any(line.startswith("Content-Type: multipart/mixed") for line in block)


@pytest.mark.asyncio
async def test_without_headers_the_message_is_unchanged():
    plain, _ = await _send()
    explicit_none, _ = await _send(headers=None)
    empty, _ = await _send(headers={})
    assert plain == explicit_none == empty
    assert not any(line.lower().startswith("list-") for line in _header_block(plain))


@pytest.mark.asyncio
async def test_a_header_off_the_list_is_refused_before_anything_is_sent():
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_response({"id": "msg1"}))
    with (
        patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)),
        pytest.raises(ValueError, match="Header not allowed"),
    ):
        await _tool().handler(
            to="guest@example.com",
            subject="Hola",
            body="Hola",
            cc=None,
            bcc=None,
            attachment_urls=None,
            message_format="text",
            email_signature=None,
            headers={"Bcc": "someone@example.com"},
        )
    mock_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_value_with_a_line_break_is_refused():
    with pytest.raises(ValueError, match="single line"):
        await _send(headers={"List-Unsubscribe": "<https://x.example>\r\nBcc: someone@example.com"})


@pytest.mark.asyncio
async def test_an_empty_value_is_skipped():
    message, _ = await _send(headers={"Reply-To": "  "})
    assert not any(line.startswith("Reply-To:") for line in _header_block(message))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("support@example.com", [("", "support@example.com")]),
        ('"Support, Team" <support@example.com>', [("Support, Team", "support@example.com")]),
        ("José <jose@example.com>", [("José", "jose@example.com")]),
        (
            "José <jose@example.com>, Zoë <zoe@example.com>",
            [("José", "jose@example.com"), ("Zoë", "zoe@example.com")],
        ),
        ("Support: José <jose@example.com>;", [("José", "jose@example.com")]),
    ],
)
async def test_reply_to_preserves_display_names_and_mailboxes(value, expected):
    message, _ = await _send(headers={"Reply-To": value})
    reply_to = Parser(policy=policy.default).parsestr(message)["Reply-To"]
    assert not reply_to.defects
    assert [(address.display_name, address.addr_spec) for address in reply_to.addresses] == expected
