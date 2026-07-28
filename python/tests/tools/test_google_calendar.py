"""Unit tests for Google Calendar tools (httpx mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.google_calendar import (
    GoogleCalendarCreateEvent,
    GoogleCalendarUpdateEvent,
    _meet_create_request,
)


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


def test_meet_create_request_is_unique_per_call():
    first = _meet_create_request()["createRequest"]
    second = _meet_create_request()["createRequest"]

    assert first["conferenceSolutionKey"] == {"type": "hangoutsMeet"}
    # A repeated requestId makes Google hand back the conference it already
    # created, which is how every event ends up sharing one room.
    assert first["requestId"] != second["requestId"]


@pytest.mark.asyncio
async def test_create_event_without_meet_sends_no_conference_data():
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_response({"id": "evt1"}))

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleCalendarCreateEvent(token=SecretStr("token"))
        out = await tool.handler(
            summary="Sync",
            start="2026-01-01T10:00:00Z",
            end="2026-01-01T10:30:00Z",
            calendar_id="primary",
            description=None,
            location=None,
            attendees=None,
            timezone="UTC",
            add_google_meet=False,
        )

    assert out["id"] == "evt1"
    call = mock_client.post.await_args
    assert "conferenceData" not in call.kwargs["json"]
    assert call.kwargs["params"] == {}


@pytest.mark.asyncio
async def test_create_event_with_meet_requests_a_new_conference():
    mock_client = MagicMock()
    mock_client.post = AsyncMock(
        return_value=_response(
            {
                "id": "evt1",
                "hangoutLink": "https://meet.google.com/abc-defg-hij",
                "conferenceData": {"status": {"statusCode": "success"}},
            }
        )
    )
    mock_client.get = AsyncMock()

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleCalendarCreateEvent(token=SecretStr("token"))
        out = await tool.handler(
            summary="Acme <> Timbal AI",
            start="2026-01-01T10:00:00Z",
            end="2026-01-01T10:30:00Z",
            calendar_id="primary",
            description=None,
            location=None,
            attendees=["prospect@acme.com"],
            timezone="Europe/Madrid",
            add_google_meet=True,
        )

    assert out["hangoutLink"] == "https://meet.google.com/abc-defg-hij"

    call = mock_client.post.await_args
    # The opt-in is a query parameter; without it Google drops conferenceData
    # from the body and the event comes back with no link at all.
    assert call.kwargs["params"]["conferenceDataVersion"] == 1
    create_request = call.kwargs["json"]["conferenceData"]["createRequest"]
    assert create_request["conferenceSolutionKey"] == {"type": "hangoutsMeet"}
    assert create_request["requestId"]

    # The link was already there, so no re-read.
    mock_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_event_polls_until_the_conference_is_minted():
    pending = _response({"id": "evt1", "conferenceData": {"status": {"statusCode": "pending"}}})
    ready = _response(
        {
            "id": "evt1",
            "hangoutLink": "https://meet.google.com/xyz-uvwx-yz",
            "conferenceData": {"status": {"statusCode": "success"}},
        }
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=pending)
    mock_client.get = AsyncMock(return_value=ready)

    with (
        patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        tool = GoogleCalendarCreateEvent(token=SecretStr("token"))
        out = await tool.handler(
            summary="Acme <> Timbal AI",
            start="2026-01-01T10:00:00Z",
            end="2026-01-01T10:30:00Z",
            calendar_id="primary",
            description=None,
            location=None,
            attendees=None,
            timezone="UTC",
            add_google_meet=True,
        )

    assert out["hangoutLink"] == "https://meet.google.com/xyz-uvwx-yz"
    assert mock_client.get.await_count == 1
    assert mock_client.get.await_args.kwargs["params"]["conferenceDataVersion"] == 1


@pytest.mark.asyncio
async def test_create_event_stops_polling_when_the_conference_failed():
    failed = _response({"id": "evt1", "conferenceData": {"status": {"statusCode": "failure"}}})

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=failed)
    mock_client.get = AsyncMock()

    with (
        patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        tool = GoogleCalendarCreateEvent(token=SecretStr("token"))
        out = await tool.handler(
            summary="Sync",
            start="2026-01-01T10:00:00Z",
            end="2026-01-01T10:30:00Z",
            calendar_id="primary",
            description=None,
            location=None,
            attendees=None,
            timezone="UTC",
            add_google_meet=True,
        )

    # The event still exists; it simply has no link. Callers decide what to do.
    assert out["id"] == "evt1"
    assert "hangoutLink" not in out
    mock_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_event_can_attach_a_conference():
    mock_client = MagicMock()
    mock_client.patch = AsyncMock(
        return_value=_response(
            {
                "id": "evt1",
                "hangoutLink": "https://meet.google.com/abc-defg-hij",
                "conferenceData": {"status": {"statusCode": "success"}},
            }
        )
    )

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleCalendarUpdateEvent(token=SecretStr("token"))
        out = await tool.handler(
            event_id="evt1",
            calendar_id="primary",
            summary=None,
            start="2026-01-02T10:00:00Z",
            end="2026-01-02T10:30:00Z",
            description=None,
            location=None,
            timezone="Europe/Madrid",
            add_google_meet=True,
        )

    assert out["hangoutLink"] == "https://meet.google.com/abc-defg-hij"
    call = mock_client.patch.await_args
    assert call.kwargs["params"]["conferenceDataVersion"] == 1
    assert call.kwargs["json"]["conferenceData"]["createRequest"]["requestId"]


@pytest.mark.asyncio
async def test_update_event_leaves_an_existing_conference_alone():
    mock_client = MagicMock()
    mock_client.patch = AsyncMock(return_value=_response({"id": "evt1"}))

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleCalendarUpdateEvent(token=SecretStr("token"))
        await tool.handler(
            event_id="evt1",
            calendar_id="primary",
            summary="Moved",
            start=None,
            end=None,
            description=None,
            location=None,
            timezone=None,
            add_google_meet=False,
        )

    call = mock_client.patch.await_args
    # PATCH leaves unspecified fields untouched, so a plain reschedule must not
    # send conferenceData — the event keeps whatever link it already had.
    assert "conferenceData" not in call.kwargs["json"]
    assert call.kwargs["params"] == {}
