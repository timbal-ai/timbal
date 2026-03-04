import base64
from typing import Annotated, Any
from urllib.parse import unquote, urlparse

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


class ReadEmails(Tool):
    name: str = "outlook_read_emails"
    description: str | None = (
        "Read emails from Outlook. Supports Microsoft Graph OData filters and content search."
    )
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _read_emails(
            folder: str = "inbox",
            top: int = 10,
            skip: int = 0,
            filter: str | None = None,
            search: str | None = None,
            select: list[str] | None = None,
            order_by: str = "receivedDateTime desc",
        ) -> Any:
            """
            folder: well-known folder name ("inbox", "sentitems", "drafts", "deleteditems", "archive")
                    or a folder ID.
            filter: OData $filter expression, e.g. "isRead eq false"
            search: OData $search query, e.g. '"subject:invoice"'
            select: list of properties to return, e.g. ["subject", "from", "receivedDateTime", "bodyPreview"]
            order_by: OData $orderby expression.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

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
                    f"{_GRAPH_API_BASE}/me/mailFolders/{folder}/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/ReadEmails"

        super().__init__(handler=_read_emails, metadata=metadata, **kwargs)


class SendEmail(Tool):
    name: str = "outlook_send_email"
    description: str | None = "Send an email using Outlook. Can automatically download and attach files from URLs using the attachments parameter."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_filename_from_url(url: str) -> str:
            """Extract filename from URL."""
            parsed = urlparse(url)
            filename = unquote(parsed.path.split('/')[-1])
            return filename or "attachment"

        async def _download_and_encode(url: str) -> str:
            """Download file from URL and return base64 encoded content."""
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return base64.b64encode(response.content).decode('utf-8')

        def _get_content_type(filename: str) -> str:
            """Return MIME type based on file extension."""
            if filename.lower().endswith('.pdf'):
                return "application/pdf"
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                return "image/jpeg"
            elif filename.lower().endswith('.png'):
                return "image/png"
            return "application/octet-stream"

        async def _send_email(
            to: list[str],
            subject: str,
            body: str,
            body_type: str = "Text",
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
            reply_to: list[str] | None = None,
            save_to_sent_items: bool = True,
            attachments: list[dict[str, Any]] | None = None,
        ) -> Any:
            """
            to: list of recipient email addresses.
            body_type: "Text" or "HTML"
            attachments: list of attachment dicts with 'name' and either:
                        - 'content_bytes': base64 encoded file content
                        - 'content_url': URL to download file from (will be auto-downloaded)
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

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
                    # Get filename from attachment or extract from URL
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
                        "contentType": _get_content_type(filename)
                    }
                    
                    # Handle content from different sources
                    if "content_bytes" in attachment:
                        attachment_obj["contentBytes"] = attachment["content_bytes"]
                    elif "content_url" in attachment:
                        attachment_obj["contentBytes"] = await _download_and_encode(attachment["content_url"])
                    elif "url" in attachment:
                        attachment_obj["contentBytes"] = await _download_and_encode(attachment["url"])
                    
                    attachment_data.append(attachment_obj)
                
                message["attachments"] = attachment_data

            async with httpx.AsyncClient() as client:
                payload = {"message": message, "saveToSentItems": save_to_sent_items}
                
                response = await client.post(
                    f"{_GRAPH_API_BASE}/me/sendMail",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                
                response.raise_for_status()
                return {"sent": True}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/SendEmail"

        super().__init__(handler=_send_email, metadata=metadata, **kwargs)


class UpdateEmail(Tool):
    name: str = "outlook_update_email"
    description: str | None = "Update email properties like read/unread status, flagged status, or move to folder."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_email(
            message_id: str,
            is_read: bool | None = None,
            flag_status: str | None = None,
            categories: list[str] | None = None,
        ) -> Any:
            """
            flag_status: "flagged", "complete", or "notFlagged"
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            body: dict[str, Any] = {}
            if is_read is not None:
                body["isRead"] = is_read
            if flag_status:
                body["flag"] = {"flagStatus": flag_status}
            if categories is not None:
                body["categories"] = categories

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_GRAPH_API_BASE}/me/messages/{message_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/UpdateEmail"

        super().__init__(handler=_update_email, metadata=metadata, **kwargs)


class CreateDraft(Tool):
    name: str = "outlook_create_draft"
    description: str | None = "Create a new email draft in Outlook."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_draft(
            subject: str,
            body: str,
            body_type: str = "Text",
            to: list[str] | None = None,
            cc: list[str] | None = None,
            bcc: list[str] | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

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
                    f"{_GRAPH_API_BASE}/me/messages",
                    headers={"Authorization": f"Bearer {token}"},
                    json=message,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/CreateDraft"

        super().__init__(handler=_create_draft, metadata=metadata, **kwargs)


class ForwardEmail(Tool):
    name: str = "outlook_forward_email"
    description: str | None = "Forward an email to specified recipients."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _forward_email(
            message_id: str,
            to: list[str],
            comment: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            body: dict[str, Any] = {
                "toRecipients": [{"emailAddress": {"address": e}} for e in to],
            }
            if comment:
                body["comment"] = comment

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_GRAPH_API_BASE}/me/messages/{message_id}/forward",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return {"forwarded": True}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/ForwardEmail"

        super().__init__(handler=_forward_email, metadata=metadata, **kwargs)


class ArchiveEmail(Tool):
    name: str = "outlook_archive_email"
    description: str | None = "Archive an email by moving it to the Archive folder."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _archive_email(message_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_GRAPH_API_BASE}/me/messages/{message_id}/move",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"destinationId": "archive"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/ArchiveEmail"

        super().__init__(handler=_archive_email, metadata=metadata, **kwargs)


class TrashEmail(Tool):
    name: str = "outlook_trash_email"
    description: str | None = "Move an email to the Trash/Deleted Items folder."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _trash_email(message_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_GRAPH_API_BASE}/me/messages/{message_id}/move",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"destinationId": "deleteditems"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/TrashEmail"

        super().__init__(handler=_trash_email, metadata=metadata, **kwargs)


class GetAttachments(Tool):
    name: str = "outlook_get_attachments"
    description: str | None = "Get attachments from an Outlook email, returning their names, content types, and base64-encoded content."
    integration: Annotated[str, Integration("outlook")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_attachments(
            message_id: str,
            include_content: bool = True,
        ) -> Any:
            """
            include_content: if True, returns the full base64-encoded contentBytes for each attachment.
                             Set to False to only retrieve metadata (name, size, contentType).
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            params: dict[str, Any] = {}
            if not include_content:
                params["$select"] = "id,name,contentType,size,isInline,lastModifiedDateTime"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_GRAPH_API_BASE}/me/messages/{message_id}/attachments",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Outlook/GetAttachments"

        super().__init__(handler=_get_attachments, metadata=metadata, **kwargs)
