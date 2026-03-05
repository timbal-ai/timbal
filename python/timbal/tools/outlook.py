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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
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

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/me/messages/{message_id}/move",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"destinationId": "deleteditems"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_trash_email, **kwargs)


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

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/me/messages/{message_id}/attachments",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_attachments, **kwargs)
