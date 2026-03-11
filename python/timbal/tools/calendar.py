import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Google Calendar API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("GOOGLE_CALENDAR_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Google Calendar API key not found. Set GOOGLE_CALENDAR_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )

class ListEvents(Tool):
    name: str = "google_calendar_list_events"
    description: str | None = "List events from Google Calendar for a specified time range."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_events(
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
            time_min: str | None = Field(None, description="Start time in ISO format, e.g. '2025-10-15T00:00:00Z'"),
            time_max: str | None = Field(None, description="End time in ISO format, e.g. '2025-10-22T23:59:59Z'"),
            max_results: int = Field(10, description="Maximum number of events to return."),
            q: str | None = Field(None, description="Free-text search query, e.g. 'meeting with John'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
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

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/ListEvents"

        super().__init__(handler=_list_events, metadata=metadata, **kwargs)


class CreateEvent(Tool):
    name: str = "google_calendar_create_event"
    description: str | None = "Create a new event in Google Calendar."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
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
        ) -> Any:
            api_key = await _resolve_api_key(self)
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

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/CreateEvent"

        super().__init__(handler=_create_event, metadata=metadata, **kwargs)


class UpdateEvent(Tool):
    name: str = "google_calendar_update_event"
    description: str | None = "Update an existing event in Google Calendar."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_event(
            event_id: str = Field(..., description="Event ID to update."),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
            summary: str | None = Field(None, description="Updated event title or summary."),
            start: str | None = Field(None, description="Updated start time in ISO format, e.g. '2025-10-15T14:00:00Z'"),
            end: str | None = Field(None, description="Updated end time in ISO format, e.g. '2025-10-15T15:00:00Z'"),
            description: str | None = Field(None, description="Updated event description or notes."),
            location: str | None = Field(None, description="Updated event location or venue."),
            timezone: str | None = Field(None, description="Updated timezone for the event, e.g. 'America/New_York'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
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

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/UpdateEvent"

        super().__init__(handler=_update_event, metadata=metadata, **kwargs)


class DeleteEvent(Tool):
    name: str = "google_calendar_delete_event"
    description: str | None = "Delete an event from Google Calendar."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_event(
            event_id: str = Field(..., description="Event ID to delete."),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return {"deleted": True, "event_id": event_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/DeleteEvent"

        super().__init__(handler=_delete_event, metadata=metadata, **kwargs)


class UpdateAttendeeStatus(Tool):
    name: str = "google_calendar_update_attendee_status"
    description: str | None = "Update an attendee's response status for a Google Calendar event."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_attendee_status(
            event_id: str = Field(..., description="Event ID to update."),
            attendee_email: str = Field(..., description="Email address of the attendee."),
            status: str = Field(..., description="New response status for the attendee. One of 'accepted', 'declined', 'tentative', 'needsAction'"),
            calendar_id: str = Field("primary", description="Calendar ID, e.g. 'primary' or 'user@example.com'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            headers = {"Authorization": f"Bearer {api_key}"}

            async with httpx.AsyncClient() as client:
                get_response = await client.get(
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
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
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers=headers,
                    json={"attendees": attendees},
                )
                patch_response.raise_for_status()
                return patch_response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/UpdateAttendeeStatus"

        super().__init__(handler=_update_attendee_status, metadata=metadata, **kwargs)


class CheckFreeSlots(Tool):
    name: str = "google_calendar_check_free_slots"
    description: str | None = "Check for available (free) time slots across one or more Google Calendars."
    integration: Annotated[str, Integration("google_calendar")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check_free_slots(
            time_min: str = Field(..., description="Start time in ISO format, e.g. '2025-10-15T00:00:00Z'"),
            time_max: str = Field(..., description="End time in ISO format, e.g. '2025-10-15T23:59:59Z'"),
            calendars: list[str] | None = Field(None, description="List of calendar IDs to check, e.g. ['primary', 'user@example.com']"),
            timezone: str = Field("UTC", description="Timezone for the time range, e.g. 'America/New_York'"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            calendar_ids = calendars or ["primary"]

            body: dict[str, Any] = {
                "timeMin": time_min,
                "timeMax": time_max,
                "timeZone": timezone,
                "items": [{"id": cal_id} for cal_id in calendar_ids],
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_CALENDAR_API_BASE}/freeBusy",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/CheckFreeSlots"

        super().__init__(handler=_check_free_slots, metadata=metadata, **kwargs)
