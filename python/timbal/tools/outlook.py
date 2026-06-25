import base64
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

from pydantic import Field

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://graph.microsoft.com/v1.0"


async def _resolve_token(tool: Any) -> str:
    """Resolve Outlook OAuth token from integration."""
    if not isinstance(getattr(tool, "integration", None), Integration):
        raise ValueError("Outlook integration not configured.")
    credentials = await tool.integration.resolve()
    return credentials["token"]


class OutlookReadEmails(Tool):
    name: str = "outlook_read_emails"
    description: str | None = "Read emails from Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _read_emails(
            folder: str = Field(
                "inbox",
                description='Folder name ("inbox", "sentitems", "drafts", "deleteditems", "archive") or folder ID',
            ),
            top: int = Field(10, description="Maximum number of emails to return"),
            skip: int = Field(0, description="Number of emails to skip"),
            filter: str | None = Field(None, description='OData $filter, e.g. "isRead eq false"'),
            search: str | None = Field(None, description='OData $search query, e.g. "subject:invoice"'),
            select: list[str] | None = Field(
                None, description='Properties to return, e.g. ["subject", "from", "receivedDateTime"]'
            ),
            order_by: str = Field("receivedDateTime desc", description="OData $orderby expression"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "$top": top,
                "$skip": skip,
                "$orderby": order_by,
            }
            if filter:
                params["$filter"] = filter
            if search:
                params["$search"] = search
            if select:
                params["$select"] = ",".join(select)

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_BASE_URL}/me/mailFolders/{folder}/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_read_emails, **kwargs)


class OutlookSend(Tool):
    name: str = "outlook_send"
    description: str | None = "Send an email via Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_filename_from_url(url: str) -> str:
            parsed = urlparse(url)
            filename = unquote(parsed.path.split("/")[-1])
            return filename or "attachment"

        async def _download_and_encode(url: str) -> str:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(url)
                response.raise_for_status()
                return base64.b64encode(response.content).decode("utf-8")

        def _get_content_type(filename: str) -> str:
            if filename.lower().endswith(".pdf"):
                return "application/pdf"
            elif filename.lower().endswith((".jpg", ".jpeg")):
                return "image/jpeg"
            elif filename.lower().endswith(".png"):
                return "image/png"
            return "application/octet-stream"

        async def _send_email(
            to: list[str] = Field(..., description="Recipient email addresses"),
            subject: str = Field(..., description="Email subject"),
            body: str = Field(..., description="Email body content"),
            body_type: str = Field("Text", description='"Text" or "HTML"'),
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
            reply_to: list[str] | None = None,
            save_to_sent_items: bool = True,
            attachments: list[dict[str, Any]] | None = Field(
                None, description="Attachments with name and content_bytes or content_url"
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            def _address_list(emails: list[str]) -> list[dict]:
                return [{"emailAddress": {"address": e}} for e in emails]

            message: dict[str, Any] = {
                "subject": subject,
                "body": {"contentType": body_type, "content": body},
                "toRecipients": _address_list(to),
            }
            if cc:
                message["ccRecipients"] = _address_list(cc)
            if bcc:
                message["bccRecipients"] = _address_list(bcc)
            if reply_to:
                message["replyTo"] = _address_list(reply_to)

            if attachments:
                attachment_data = []
                for attachment in attachments:
                    filename = attachment.get("name")
                    if not filename:
                        if "url" in attachment:
                            filename = await _get_filename_from_url(attachment["url"])
                        elif "content_url" in attachment:
                            filename = await _get_filename_from_url(attachment["content_url"])
                        else:
                            filename = "attachment"

                    attachment_obj = {
                        "@odata.type": "#microsoft.graph.fileAttachment",
                        "name": filename,
                        "contentType": _get_content_type(filename),
                    }

                    if "content_bytes" in attachment:
                        attachment_obj["contentBytes"] = attachment["content_bytes"]
                    elif "content_url" in attachment:
                        attachment_obj["contentBytes"] = await _download_and_encode(attachment["content_url"])
                    elif "url" in attachment:
                        attachment_obj["contentBytes"] = await _download_and_encode(attachment["url"])

                    attachment_data.append(attachment_obj)

                message["attachments"] = attachment_data

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/sendMail",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"message": message, "saveToSentItems": save_to_sent_items},
                )
                response.raise_for_status()
                return {"sent": True}

        super().__init__(handler=_send_email, **kwargs)


class OutlookUpdateEmail(Tool):
    name: str = "outlook_update_email"
    description: str | None = "Update email properties in Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_email(
            message_id: str = Field(..., description="ID of the email to update"),
            is_read: bool | None = Field(None, description="Mark as read or unread"),
            flag_status: str | None = Field(None, description='"flagged", "complete", or "notFlagged"'),
            categories: list[str] | None = Field(None, description="Categories to assign"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {}
            if is_read is not None:
                body["isRead"] = is_read
            if flag_status:
                body["flag"] = {"flagStatus": flag_status}
            if categories is not None:
                body["categories"] = categories

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.patch(
                    f"{_BASE_URL}/me/messages/{message_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_email, **kwargs)


class OutlookCreateDraft(Tool):
    name: str = "outlook_create_draft"
    description: str | None = "Create an email draft in Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_draft(
            subject: str = Field(..., description="Email subject"),
            body: str = Field(..., description="Email body content"),
            body_type: str = Field("Text", description='"Text" or "HTML"'),
            to: list[str] | None = Field(None, description="Recipient email addresses"),
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            def _address_list(emails: list[str]) -> list[dict]:
                return [{"emailAddress": {"address": e}} for e in emails]

            message: dict[str, Any] = {
                "subject": subject,
                "body": {"contentType": body_type, "content": body},
            }
            if to:
                message["toRecipients"] = _address_list(to)
            if cc:
                message["ccRecipients"] = _address_list(cc)
            if bcc:
                message["bccRecipients"] = _address_list(bcc)

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    json=message,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_draft, **kwargs)


class OutlookForward(Tool):
    name: str = "outlook_forward"
    description: str | None = "Forward an email in Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _forward_email(
            message_id: str = Field(..., description="ID of the email to forward"),
            to: list[str] = Field(..., description="Recipient email addresses"),
            comment: str | None = Field(None, description="Comment to include with the forwarded email"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "toRecipients": [{"emailAddress": {"address": e}} for e in to],
            }
            if comment:
                body["comment"] = comment

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/messages/{message_id}/forward",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return {"forwarded": True}

        super().__init__(handler=_forward_email, **kwargs)


class OutlookArchive(Tool):
    name: str = "outlook_archive"
    description: str | None = "Archive an email in Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _archive_email(
            message_id: str = Field(..., description="ID of the email to archive"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/messages/{message_id}/move",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"destinationId": "archive"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_archive_email, **kwargs)


class OutlookTrash(Tool):
    name: str = "outlook_trash"
    description: str | None = "Move an email to trash in Outlook."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _trash_email(
            message_id: str = Field(..., description="ID of the email to trash"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/messages/{message_id}/move",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"destinationId": "deleteditems"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_trash_email, **kwargs)


class OutlookListEvents(Tool):
    name: str = "outlook_list_events"
    description: str | None = (
        "List or search calendar events in Outlook. Supports date range, free-text keyword search, "
        "and OData filters (e.g. by attendee, organizer, importance)."
    )
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_events(
            start_datetime: str | None = Field(
                None,
                description=(
                    "Start of the date range in ISO 8601, e.g. '2026-06-03T00:00:00'. "
                    "When both start_datetime and end_datetime are provided (and no search), "
                    "calendarView is used and recurring events are expanded into individual occurrences."
                ),
            ),
            end_datetime: str | None = Field(
                None,
                description="End of the date range in ISO 8601, e.g. '2026-06-10T23:59:59'.",
            ),
            search: str | None = Field(
                None,
                description=(
                    'Free-text keyword search applied to subject/body/attendees, e.g. "project review". '
                    "Forces use of /me/events (recurrences are NOT expanded)."
                ),
            ),
            filter: str | None = Field(
                None,
                description=(
                    "OData $filter expression. Examples: "
                    "\"organizer/emailAddress/address eq 'alice@example.com'\", "
                    "\"attendees/any(a:a/emailAddress/address eq 'bob@example.com')\", "
                    "\"importance eq 'high'\"."
                ),
            ),
            calendar_id: str | None = Field(
                None,
                description="Calendar ID. If omitted, the user's default calendar is used.",
            ),
            top: int = Field(25, description="Maximum number of events to return."),
            skip: int = Field(0, description="Number of events to skip."),
            order_by: str = Field("start/dateTime asc", description="OData $orderby expression."),
            select: list[str] | None = Field(
                None,
                description='Properties to return, e.g. ["subject", "start", "end", "attendees", "organizer"].',
            ),
            timezone: str | None = Field(
                None,
                description=(
                    "Preferred timezone for returned event times, e.g. 'Europe/Madrid'. "
                    "Sent via the Prefer: outlook.timezone header. Defaults to the mailbox timezone."
                ),
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            base = f"{_BASE_URL}/me"
            if calendar_id:
                base = f"{base}/calendars/{calendar_id}"

            use_calendar_view = bool(start_datetime and end_datetime and not search)
            if use_calendar_view:
                url = f"{base}/calendarView"
                params: dict[str, Any] = {
                    "startDateTime": start_datetime,
                    "endDateTime": end_datetime,
                    "$top": top,
                    "$skip": skip,
                    "$orderby": order_by,
                }
            else:
                url = f"{base}/events"
                params = {"$top": top, "$skip": skip, "$orderby": order_by}
                date_filters: list[str] = []
                if start_datetime:
                    date_filters.append(f"end/dateTime ge '{start_datetime}'")
                if end_datetime:
                    date_filters.append(f"start/dateTime le '{end_datetime}'")
                combined_filter = " and ".join([*date_filters, filter] if filter else date_filters)
                if combined_filter:
                    params["$filter"] = combined_filter
                if search:
                    params["$search"] = f'"{search}"'

            if filter and use_calendar_view:
                params["$filter"] = filter
            if select:
                params["$select"] = ",".join(select)

            headers = {"Authorization": f"Bearer {token}"}
            if timezone:
                headers["Prefer"] = f'outlook.timezone="{timezone}"'

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_events, **kwargs)


class OutlookGetEvent(Tool):
    name: str = "outlook_get_event"
    description: str | None = "Read the full details of a single Outlook calendar event by ID."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_event(
            event_id: str = Field(..., description="ID of the event to retrieve."),
            calendar_id: str | None = Field(
                None,
                description="Calendar ID. If omitted, the user's default calendar is used.",
            ),
            select: list[str] | None = Field(
                None,
                description='Properties to return. Defaults to all. Example: ["subject", "body", "attendees"].',
            ),
            timezone: str | None = Field(
                None,
                description=(
                    "Preferred timezone for returned event times, e.g. 'Europe/Madrid'. "
                    "Sent via the Prefer: outlook.timezone header."
                ),
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            base = f"{_BASE_URL}/me"
            if calendar_id:
                base = f"{base}/calendars/{calendar_id}"
            url = f"{base}/events/{event_id}"

            params: dict[str, Any] = {}
            if select:
                params["$select"] = ",".join(select)

            headers = {"Authorization": f"Bearer {token}"}
            if timezone:
                headers["Prefer"] = f'outlook.timezone="{timezone}"'

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_event, **kwargs)


class OutlookGetSchedule(Tool):
    name: str = "outlook_get_schedule"
    description: str | None = (
        "Get free/busy availability for one or more users over a time range, useful for scheduling. "
        "Wraps Microsoft Graph's /me/calendar/getSchedule."
    )
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_schedule(
            schedules: list[str] = Field(
                ...,
                description="Email addresses to check, e.g. ['alice@example.com', 'bob@example.com'].",
            ),
            start_datetime: str = Field(
                ...,
                description="Start of the window in ISO 8601, e.g. '2026-06-03T09:00:00'.",
            ),
            end_datetime: str = Field(
                ...,
                description="End of the window in ISO 8601, e.g. '2026-06-03T18:00:00'.",
            ),
            timezone: str = Field("UTC", description="Timezone for start_datetime/end_datetime, e.g. 'Europe/Madrid'."),
            availability_view_interval: int = Field(
                30,
                description="Granularity of the availability view, in minutes (5-1440). Defaults to 30.",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "schedules": schedules,
                "startTime": {"dateTime": start_datetime, "timeZone": timezone},
                "endTime": {"dateTime": end_datetime, "timeZone": timezone},
                "availabilityViewInterval": availability_view_interval,
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/calendar/getSchedule",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_schedule, **kwargs)


class OutlookFindMeetingTimes(Tool):
    name: str = "outlook_find_meeting_times"
    description: str | None = (
        "Suggest meeting times that work for a set of attendees, given a duration and time constraints. "
        "Wraps Microsoft Graph's /me/findMeetingTimes."
    )
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_meeting_times(
            attendees: list[str] = Field(
                ...,
                description="Attendee email addresses, e.g. ['alice@example.com', 'bob@example.com'].",
            ),
            meeting_duration: str = Field(
                "PT30M",
                description="Meeting duration as an ISO 8601 duration, e.g. 'PT30M' (30 min) or 'PT1H' (1 hour).",
            ),
            start_datetime: str | None = Field(
                None,
                description="Earliest acceptable start time in ISO 8601, e.g. '2026-06-03T09:00:00'.",
            ),
            end_datetime: str | None = Field(
                None,
                description="Latest acceptable end time in ISO 8601, e.g. '2026-06-10T18:00:00'.",
            ),
            timezone: str = Field("UTC", description="Timezone for start_datetime/end_datetime, e.g. 'Europe/Madrid'."),
            max_candidates: int = Field(20, description="Maximum number of suggested meeting time candidates."),
            minimum_attendee_percentage: float = Field(
                100.0,
                description="Minimum percentage of attendees that must be available (0-100).",
            ),
            is_organizer_optional: bool = Field(False, description="Whether the organizer can be absent."),
            return_suggestion_reasons: bool = Field(
                True,
                description="Include human-readable reasoning for each suggested time.",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "attendees": [
                    {"type": "required", "emailAddress": {"address": email}} for email in attendees
                ],
                "meetingDuration": meeting_duration,
                "maxCandidates": max_candidates,
                "minimumAttendeePercentage": minimum_attendee_percentage,
                "isOrganizerOptional": is_organizer_optional,
                "returnSuggestionReasons": return_suggestion_reasons,
            }
            if start_datetime and end_datetime:
                body["timeConstraint"] = {
                    "timeSlots": [
                        {
                            "start": {"dateTime": start_datetime, "timeZone": timezone},
                            "end": {"dateTime": end_datetime, "timeZone": timezone},
                        }
                    ]
                }

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/me/findMeetingTimes",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_find_meeting_times, **kwargs)


def _format_attendees(emails: list[str] | None, attendee_type: str) -> list[dict[str, Any]]:
    if not emails:
        return []
    return [
        {"emailAddress": {"address": email}, "type": attendee_type}
        for email in emails
    ]


class OutlookCreateEvent(Tool):
    name: str = "outlook_create_event"
    description: str | None = (
        "Create a new calendar event in Outlook and (optionally) invite attendees. "
        "Supports Microsoft Teams online meetings."
    )
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_event(
            subject: str = Field(..., description="Event title."),
            start_datetime: str = Field(
                ...,
                description="Start time in ISO 8601, e.g. '2026-06-05T14:00:00'.",
            ),
            end_datetime: str = Field(
                ...,
                description="End time in ISO 8601, e.g. '2026-06-05T15:00:00'.",
            ),
            timezone: str = Field("UTC", description="Timezone for start/end, e.g. 'Europe/Madrid'."),
            attendees: list[str] | None = Field(
                None, description="Required attendee email addresses."
            ),
            optional_attendees: list[str] | None = Field(
                None, description="Optional attendee email addresses."
            ),
            location: str | None = Field(None, description="Event location, e.g. 'Conference Room A'."),
            body: str | None = Field(None, description="Event description / body content."),
            body_type: str = Field("Text", description='"Text" or "HTML".'),
            is_online_meeting: bool = Field(
                False,
                description="If true, create a Microsoft Teams meeting and include the join link in the event.",
            ),
            calendar_id: str | None = Field(
                None,
                description="Calendar ID. If omitted, the user's default calendar is used.",
            ),
            send_invitations: bool = Field(
                True,
                description=(
                    "If true (default), Outlook sends invitations to attendees. "
                    "If false, the event is created without notifying anyone."
                ),
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            event: dict[str, Any] = {
                "subject": subject,
                "start": {"dateTime": start_datetime, "timeZone": timezone},
                "end": {"dateTime": end_datetime, "timeZone": timezone},
            }
            if body is not None:
                event["body"] = {"contentType": body_type, "content": body}
            if location:
                event["location"] = {"displayName": location}
            invited = [
                *_format_attendees(attendees, "required"),
                *_format_attendees(optional_attendees, "optional"),
            ]
            if invited:
                event["attendees"] = invited
            if is_online_meeting:
                event["isOnlineMeeting"] = True
                event["onlineMeetingProvider"] = "teamsForBusiness"
            if not send_invitations:
                # Microsoft Graph: this flag suppresses invitation emails on creation.
                event["responseRequested"] = False

            base = f"{_BASE_URL}/me"
            if calendar_id:
                base = f"{base}/calendars/{calendar_id}"

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{base}/events",
                    headers={"Authorization": f"Bearer {token}"},
                    json=event,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_event, **kwargs)


class OutlookManageEvent(Tool):
    name: str = "outlook_manage_event"
    description: str | None = (
        "Perform an action on an existing Outlook calendar event: update fields, cancel (notifies attendees), "
        "delete (no notification), or respond to an invitation (accept / tentative / decline)."
    )
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _manage_event(
            event_id: str = Field(..., description="ID of the event to act on."),
            action: str = Field(
                ...,
                description=(
                    "Action to perform. One of: "
                    "'update' (patch event fields — only fields you pass are changed; passing 'attendees' "
                    "or 'optional_attendees' REPLACES the entire attendee list), "
                    "'cancel' (organizer-only; sends a cancellation notice to attendees), "
                    "'delete' (hard delete, no notifications), "
                    "'accept' / 'tentative' / 'decline' (respond to an invitation)."
                ),
            ),
            calendar_id: str | None = Field(
                None,
                description="Calendar ID. If omitted, the user's default calendar is used. Ignored for response actions.",
            ),
            # --- update params ---
            subject: str | None = Field(None, description="[update] New event title."),
            start_datetime: str | None = Field(
                None, description="[update] New start time in ISO 8601, e.g. '2026-06-05T14:00:00'."
            ),
            end_datetime: str | None = Field(
                None, description="[update] New end time in ISO 8601, e.g. '2026-06-05T15:00:00'."
            ),
            timezone: str | None = Field(
                None, description="[update] Timezone for start/end if changed, e.g. 'Europe/Madrid'."
            ),
            location: str | None = Field(None, description="[update] New event location."),
            body: str | None = Field(None, description="[update] New event description / body."),
            body_type: str = Field("Text", description='[update] "Text" or "HTML" for body.'),
            attendees: list[str] | None = Field(
                None,
                description=(
                    "[update] Full list of required attendee emails. "
                    "REPLACES the existing list — include everyone who should stay invited."
                ),
            ),
            optional_attendees: list[str] | None = Field(
                None,
                description="[update] Full list of optional attendee emails. REPLACES the existing optional list.",
            ),
            is_online_meeting: bool | None = Field(
                None, description="[update] Toggle Microsoft Teams online meeting on/off."
            ),
            # --- cancel / respond params ---
            comment: str | None = Field(
                None,
                description="[cancel / accept / tentative / decline] Optional message included with the action.",
            ),
            send_response: bool = Field(
                True,
                description="[accept / tentative / decline] Whether to send a response back to the organizer.",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            valid_actions = {"update", "cancel", "delete", "accept", "tentative", "decline"}
            if action not in valid_actions:
                raise ValueError(f"Invalid action '{action}'. Must be one of {sorted(valid_actions)}.")

            base = f"{_BASE_URL}/me"
            if calendar_id and action in {"update", "cancel", "delete"}:
                base = f"{base}/calendars/{calendar_id}"
            event_url = f"{base}/events/{event_id}"
            headers = {"Authorization": f"Bearer {token}"}

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                if action == "update":
                    body_payload: dict[str, Any] = {}
                    if subject is not None:
                        body_payload["subject"] = subject
                    if start_datetime is not None:
                        body_payload["start"] = {"dateTime": start_datetime, "timeZone": timezone or "UTC"}
                    if end_datetime is not None:
                        body_payload["end"] = {"dateTime": end_datetime, "timeZone": timezone or "UTC"}
                    if location is not None:
                        body_payload["location"] = {"displayName": location}
                    if body is not None:
                        body_payload["body"] = {"contentType": body_type, "content": body}
                    if attendees is not None or optional_attendees is not None:
                        body_payload["attendees"] = [
                            *_format_attendees(attendees, "required"),
                            *_format_attendees(optional_attendees, "optional"),
                        ]
                    if is_online_meeting is not None:
                        body_payload["isOnlineMeeting"] = is_online_meeting
                        if is_online_meeting:
                            body_payload["onlineMeetingProvider"] = "teamsForBusiness"

                    if not body_payload:
                        raise ValueError("'update' action requires at least one field to change.")

                    response = await client.patch(event_url, headers=headers, json=body_payload)
                    response.raise_for_status()
                    return response.json()

                if action == "cancel":
                    payload: dict[str, Any] = {}
                    if comment:
                        payload["comment"] = comment
                    response = await client.post(f"{event_url}/cancel", headers=headers, json=payload)
                    response.raise_for_status()
                    return {"cancelled": True, "event_id": event_id}

                if action == "delete":
                    response = await client.delete(event_url, headers=headers)
                    response.raise_for_status()
                    return {"deleted": True, "event_id": event_id}

                # respond actions
                graph_action = {
                    "accept": "accept",
                    "tentative": "tentativelyAccept",
                    "decline": "decline",
                }[action]
                payload = {"sendResponse": send_response}
                if comment:
                    payload["comment"] = comment
                response = await client.post(
                    f"{_BASE_URL}/me/events/{event_id}/{graph_action}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return {"responded": True, "action": action, "event_id": event_id}

        super().__init__(handler=_manage_event, **kwargs)


class OutlookGetAttachments(Tool):
    name: str = "outlook_get_attachments"
    description: str | None = "Get attachments from an Outlook email."
    integration: Annotated[str, Integration("outlook")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_attachments(
            message_id: str = Field(..., description="ID of the email"),
            include_content: bool = Field(True, description="Include base64-encoded content (False for metadata only)"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if not include_content:
                params["$select"] = "id,name,contentType,size,isInline,lastModifiedDateTime"

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_BASE_URL}/me/messages/{message_id}/attachments",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_attachments, **kwargs)
