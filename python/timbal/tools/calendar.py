from typing import Annotated, Any

import httpx
from pydantic import BaseModel

from ..core.tool import Tool
from ..platform.integrations import Integration

_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"

class ListEvents(Tool):
    integration: Annotated[str, Integration("google_calendar")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_events(
            calendar_id: str = "primary",
            time_min: str | None = None,
            time_max: str | None = None,
            max_results: int = 10,
            q: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/ListEvents"

        super().__init__(
            name="google_calendar_list_events",
            description="List events from Google Calendar for a specified time range.",
            handler=_list_events,
            metadata=metadata,
            **kwargs,
        )


class CreateEvent(Tool):
    integration: Annotated[str, Integration("google_calendar")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_event(
            summary: str,
            start: str,
            end: str,
            calendar_id: str = "primary",
            description: str | None = None,
            location: str | None = None,
            attendees: list[str] | None = None,
            timezone: str = "UTC",
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/CreateEvent"

        super().__init__(
            name="google_calendar_create_event",
            description="Create a new event in Google Calendar.",
            handler=_create_event,
            metadata=metadata,
            **kwargs,
        )


class UpdateEvent(Tool):
    integration: Annotated[str, Integration("google_calendar")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_event(
            event_id: str,
            calendar_id: str = "primary",
            summary: str | None = None,
            start: str | None = None,
            end: str | None = None,
            description: str | None = None,
            location: str | None = None,
            timezone: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/UpdateEvent"

        super().__init__(
            name="google_calendar_update_event",
            description="Update an existing event in Google Calendar.",
            handler=_update_event,
            metadata=metadata,
            **kwargs,
        )


class DeleteEvent(Tool):
    integration: Annotated[str, Integration("google_calendar")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_event(
            event_id: str,
            calendar_id: str = "primary",
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_CALENDAR_API_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "event_id": event_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/DeleteEvent"

        super().__init__(
            name="google_calendar_delete_event",
            description="Delete an event from Google Calendar.",
            handler=_delete_event,
            metadata=metadata,
            **kwargs,
        )


class UpdateAttendeeStatus(Tool):
    integration: Annotated[str, Integration("google_calendar")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_attendee_status(
            event_id: str,
            attendee_email: str,
            status: str,
            calendar_id: str = "primary",
        ) -> Any:
            """
            status: one of "accepted", "declined", "tentative", "needsAction"
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            headers = {"Authorization": f"Bearer {token}"}

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

        super().__init__(
            name="google_calendar_update_attendee_status",
            description="Update an attendee's response status for a Google Calendar event.",
            handler=_update_attendee_status,
            metadata=metadata,
            **kwargs,
        )


class CheckFreeSlots(Tool):
    integration: Annotated[str, Integration("google_calendar")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check_free_slots(
            time_min: str,
            time_max: str,
            calendars: list[str] | None = None,
            timezone: str = "UTC",
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleCalendar/CheckFreeSlots"

        super().__init__(
            name="google_calendar_check_free_slots",
            description="Check for available (free) time slots across one or more Google Calendars.",
            handler=_check_free_slots,
            metadata=metadata,
            **kwargs,
        )
