import asyncio
import uuid
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"

# How long to keep re-reading an event whose conference is still being minted.
_CONFERENCE_POLL_DELAYS = (0.5, 1.0, 2.0)


def _meet_create_request() -> dict[str, Any]:
    """Ask Google for a brand new Meet room.

    `requestId` has to differ per call: Google treats a repeated id as a retry of
    the same request and hands back the conference it already made, so a constant
    value would quietly share one room across every event.
    """
    return {
        "createRequest": {
            "requestId": str(uuid.uuid4()),
            "conferenceSolutionKey": {"type": "hangoutsMeet"},
        }
    }


async def _await_conference(
    client: Any,
    token: str,
    calendar_id: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Re-read an event until Google has finished attaching its conference.

    Conferences are created asynchronously, so `events.insert` can answer with
    `conferenceData.status.statusCode == "pending"` and no `hangoutLink` yet. The
    link is the whole point of asking for a conference — it goes in front of a
    human — so it is worth a few cheap reads rather than returning an event that
    nobody can join.
    """
    event_id = event.get("id")
    if not event_id:
        return event

    for delay in _CONFERENCE_POLL_DELAYS:
        status = ((event.get("conferenceData") or {}).get("status") or {}).get("statusCode")
        # Only `pending` will change. No status at all means Google never took the
        # request — polling a calendar that cannot host Meet just wastes seconds.
        if event.get("hangoutLink") or status != "pending":
            break

        await asyncio.sleep(delay)
        response = await client.get(
            f"{_CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {token}"},
            params={"conferenceDataVersion": 1},
        )
        response.raise_for_status()
        event = response.json()

    return event


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        token = credentials.get("token")
        if token:
            return token
        raise ValueError("Integration credentials did not include a token.")
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError("Google Calendar credentials not found. Configure an integration or pass token.")


class GoogleCalendarListEvents(Tool):
    name: str = "google_calendar_list_events"
    description: str | None = "List events from Google Calendar for a specified time range."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_events(
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
            time_min: str | None = Field(None, description="Start time in ISO format, e.g. '2025-10-15T00:00:00Z'"),
            time_max: str | None = Field(None, description="End time in ISO format, e.g. '2025-10-22T23:59:59Z'"),
            max_results: int = Field(10, description="Maximum number of events to return."),
            q: str | None = Field(None, description="Free-text search query, e.g. 'meeting with John'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "maxResults": max_results,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if time_min:
                params["timeMin"] = time_min
            if time_max:
                params["timeMax"] = time_max
            if q:
                params["q"] = q

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CALENDAR_BASE}/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_events, **kwargs)


class GoogleCalendarCreateEvent(Tool):
    name: str = "google_calendar_create_event"
    description: str | None = "Create a new event in Google Calendar, optionally with its own Google Meet link."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_event(
            summary: str = Field(..., description="Event title or summary."),
            start: str = Field(..., description="Start time in ISO format, e.g. '2025-10-15T14:00:00Z'"),
            end: str = Field(..., description="End time in ISO format, e.g. '2025-10-15T15:00:00Z'"),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
            description: str | None = Field(None, description="Event description or notes."),
            location: str | None = Field(None, description="Event location or venue."),
            attendees: list[str] | None = Field(None, description="List of attendee email addresses."),
            timezone: str = Field("UTC", description="Timezone for the event, e.g. 'America/New_York'"),
            add_google_meet: bool = Field(
                False,
                description=(
                    "If true, create a Google Meet conference for this event and return its join "
                    "link in 'hangoutLink'. Each event gets its own room."
                ),
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "summary": summary,
                "start": {"dateTime": start, "timeZone": timezone},
                "end": {"dateTime": end, "timeZone": timezone},
            }
            if description:
                body["description"] = description
            if location:
                body["location"] = location
            if attendees:
                body["attendees"] = [{"email": email} for email in attendees]

            # Conference data is ignored unless the request opts into it, and the
            # opt-in is a query parameter rather than a body field. Both or neither.
            params: dict[str, Any] = {}
            if add_google_meet:
                body["conferenceData"] = _meet_create_request()
                params["conferenceDataVersion"] = 1

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CALENDAR_BASE}/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=body,
                )
                response.raise_for_status()
                event = response.json()

                if add_google_meet:
                    event = await _await_conference(client, token, calendar_id, event)

                return event

        super().__init__(handler=_create_event, **kwargs)


class GoogleCalendarUpdateEvent(Tool):
    name: str = "google_calendar_update_event"
    description: str | None = "Update an existing event in Google Calendar, optionally attaching a Google Meet link."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_event(
            event_id: str = Field(..., description="Event ID to update."),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
            summary: str | None = Field(None, description="Updated event title or summary."),
            start: str | None = Field(
                None, description="Updated start time in ISO format, e.g. '2025-10-15T14:00:00Z'"
            ),
            end: str | None = Field(None, description="Updated end time in ISO format, e.g. '2025-10-15T15:00:00Z'"),
            description: str | None = Field(None, description="Updated event description or notes."),
            location: str | None = Field(None, description="Updated event location or venue."),
            timezone: str | None = Field(None, description="Updated timezone for the event, e.g. 'America/New_York'"),
            add_google_meet: bool = Field(
                False,
                description=(
                    "If true, attach a Google Meet conference to this event and return its join "
                    "link in 'hangoutLink'. Events that already have one keep it."
                ),
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {}
            if summary:
                body["summary"] = summary
            if description:
                body["description"] = description
            if location:
                body["location"] = location
            if start:
                body["start"] = {"dateTime": start, "timeZone": timezone or "UTC"}
            if end:
                body["end"] = {"dateTime": end, "timeZone": timezone or "UTC"}

            params: dict[str, Any] = {}
            if add_google_meet:
                body["conferenceData"] = _meet_create_request()
                params["conferenceDataVersion"] = 1

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.patch(
                    f"{_CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=body,
                )
                response.raise_for_status()
                event = response.json()

                if add_google_meet:
                    event = await _await_conference(client, token, calendar_id, event)

                return event

        super().__init__(handler=_update_event, **kwargs)


class GoogleCalendarDeleteEvent(Tool):
    name: str = "google_calendar_delete_event"
    description: str | None = "Delete an event from Google Calendar."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_event(
            event_id: str = Field(..., description="Event ID to delete."),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.delete(
                    f"{_CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "event_id": event_id}

        super().__init__(handler=_delete_event, **kwargs)


class GoogleCalendarUpdateAttendeeStatus(Tool):
    name: str = "google_calendar_update_attendee_status"
    description: str | None = "Update an attendee's response status for a Google Calendar event."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_attendee_status(
            event_id: str = Field(..., description="Event ID to update."),
            attendee_email: str = Field(..., description="Email address of the attendee."),
            status: str = Field(
                ...,
                description="New response status for the attendee. One of 'accepted', 'declined', 'tentative', 'needsAction'",
            ),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            headers = {"Authorization": f"Bearer {token}"}

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                get_response = await client.get(
                    f"{_CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers=headers,
                )
                get_response.raise_for_status()
                event = get_response.json()

                attendees: list[dict[str, Any]] = event.get("attendees", [])
                for attendee in attendees:
                    if attendee.get("email") == attendee_email:
                        attendee["responseStatus"] = status
                        break

                patch_response = await client.patch(
                    f"{_CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers=headers,
                    json={"attendees": attendees},
                )
                patch_response.raise_for_status()
                return patch_response.json()

        super().__init__(handler=_update_attendee_status, **kwargs)


class GoogleCalendarCheckFreeSlots(Tool):
    name: str = "google_calendar_check_free_slots"
    description: str | None = "Check for available (free) time slots across one or more Google Calendars."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check_free_slots(
            time_min: str = Field(..., description="Start time in ISO format, e.g. '2025-10-15T00:00:00Z'"),
            time_max: str = Field(..., description="End time in ISO format, e.g. '2025-10-15T23:59:59Z'"),
            calendars: list[str] | None = Field(
                None, description="List of calendar IDs to check, e.g. ['primary', 'user@example.com']"
            ),
            timezone: str = Field("UTC", description="Timezone for the time range, e.g. 'America/New_York'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            calendar_ids = calendars or ["primary"]

            body: dict[str, Any] = {
                "timeMin": time_min,
                "timeMax": time_max,
                "timeZone": timezone,
                "items": [{"id": cal_id} for cal_id in calendar_ids],
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CALENDAR_BASE}/freeBusy",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_check_free_slots, **kwargs)
