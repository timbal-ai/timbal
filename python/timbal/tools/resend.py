import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.resend.com"
_USER_AGENT = "timbal-resend-tools/1.0"


async def _resolve_api_key(*, integration: Any = None, api_key: SecretStr | None = None) -> str:
    """Resolve Resend API key from integration, explicit field, or env var."""
    from ._creds import resolve_api_key

    return await resolve_api_key(
        env_var="RESEND_API_KEY",
        provider_name="Resend",
        integration=integration,
        api_key=api_key,
    )


def _headers(api_key: str, *, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _resolve_default_from(tool: Any) -> str | None:
    if tool.default_from:
        return tool.default_from
    return os.getenv("RESEND_FROM_EMAIL") or None


def _resolve_default_reply_to(tool: Any) -> list[str] | None:
    if tool.default_reply_to is not None:
        return tool.default_reply_to
    env = os.getenv("RESEND_REPLY_TO")
    if env:
        return [address.strip() for address in env.split(",") if address.strip()]
    return None


def _config_fields(tool: Any) -> dict[str, Any]:
    return tool._annotate_config(
        {
            "integration": tool.integration,
            "api_key": tool.api_key,
            "default_from": tool.default_from,
            "default_reply_to": tool.default_reply_to,
        }
    )


def _build_send_payload(
    *,
    from_address: str,
    to: list[str],
    subject: str,
    html: str | None,
    text: str | None,
    cc: list[str] | None,
    bcc: list[str] | None,
    reply_to: list[str] | None,
    scheduled_at: str | None,
    headers: dict[str, str] | None,
    attachments: list[dict] | None,
    tags: list[dict] | None,
    topic_id: str | None,
    template: dict | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "from": from_address,
        "to": to,
        "subject": subject,
    }
    if html is not None:
        payload["html"] = html
    if text is not None:
        payload["text"] = text
    if cc:
        payload["cc"] = cc
    if bcc:
        payload["bcc"] = bcc
    if reply_to:
        payload["reply_to"] = reply_to
    if scheduled_at is not None:
        payload["scheduled_at"] = scheduled_at
    if headers:
        payload["headers"] = headers
    if attachments:
        payload["attachments"] = attachments
    if tags:
        payload["tags"] = tags
    if topic_id is not None:
        payload["topic_id"] = topic_id
    if template is not None:
        payload["template"] = template
    return payload


class ResendSendEmail(Tool):
    name: str = "resend_send_email"
    description: str | None = (
        "Send an email via Resend. Supports HTML, plain text, attachments, scheduling, tags, "
        "topics, and published templates."
    )
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _send_email(
            to: list[str] = Field(..., description="Recipient email addresses (max 50)"),
            subject: str = Field(..., description="Email subject"),
            from_: str | None = Field(None, description='Sender address, e.g. "Acme <onboarding@example.com>"'),
            html: str | None = Field(None, description="HTML body"),
            text: str | None = Field(None, description="Plain text body"),
            cc: list[str] | None = Field(None, description="Cc recipient addresses"),
            bcc: list[str] | None = Field(None, description="Bcc recipient addresses"),
            reply_to: list[str] | None = Field(None, description="Reply-to addresses"),
            scheduled_at: str | None = Field(
                None,
                description="Schedule send time (natural language or ISO 8601, e.g. 2026-08-05T11:52:01.858Z)",
            ),
            headers: dict[str, str] | None = Field(None, description="Custom email headers"),
            attachments: list[dict] | None = Field(
                None,
                description=(
                    "Attachments as dicts with filename/content/path/content_type. "
                    "Use content_id for inline images."
                ),
            ),
            tags: list[dict] | None = Field(
                None,
                description='Tags as list of {"name": "...", "value": "..."} objects',
            ),
            topic_id: str | None = Field(None, description="Topic ID for subscription-aware sending"),
            template: dict | None = Field(
                None,
                description='Published template: {"id": "...", "variables": {...}}',
            ),
            idempotency_key: str | None = Field(
                None,
                description="Unique key to prevent duplicate sends (expires after 24 hours, max 256 chars)",
            ),
        ) -> dict:
            from_address = from_ or _resolve_default_from(self)
            if not from_address:
                raise ValueError("Sender address required: pass from_ or set default_from / RESEND_FROM_EMAIL.")
            if html is None and text is None and template is None:
                raise ValueError("At least one of html, text, or template is required.")

            effective_reply_to = reply_to if reply_to is not None else _resolve_default_reply_to(self)
            payload = _build_send_payload(
                from_address=from_address,
                to=to,
                subject=subject,
                html=html,
                text=text,
                cc=cc,
                bcc=bcc,
                reply_to=effective_reply_to,
                scheduled_at=scheduled_at,
                headers=headers,
                attachments=attachments,
                tags=tags,
                topic_id=topic_id,
                template=template,
            )

            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/emails",
                    headers=_headers(key, idempotency_key=idempotency_key),
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_send_email, **kwargs)


class ResendSendBatch(Tool):
    name: str = "resend_send_batch"
    description: str | None = (
        "Send up to 100 emails in one API call via Resend. Batch items do not support attachments or scheduled_at."
    )
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _send_batch(
            emails: list[dict] = Field(
                ...,
                description=(
                    "List of email objects (max 100). Each may include from, to, subject, html, text, cc, bcc, "
                    "reply_to, headers, tags, topic_id, template. Attachments and scheduled_at are not supported."
                ),
            ),
            idempotency_key: str | None = Field(None, description="Unique key to prevent duplicate batch sends"),
        ) -> dict:
            if not emails:
                raise ValueError("emails must contain at least one item.")
            if len(emails) > 100:
                raise ValueError("emails must contain at most 100 items.")

            default_from = _resolve_default_from(self)
            default_reply_to = _resolve_default_reply_to(self)
            normalized: list[dict[str, Any]] = []
            for index, item in enumerate(emails):
                email = dict(item)
                if not email.get("from"):
                    if not default_from:
                        raise ValueError(
                            f"Email at index {index} missing 'from' and no default_from / RESEND_FROM_EMAIL set."
                        )
                    email["from"] = default_from
                if "reply_to" not in email and default_reply_to:
                    email["reply_to"] = default_reply_to
                normalized.append(email)

            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/emails/batch",
                    headers=_headers(key, idempotency_key=idempotency_key),
                    json=normalized,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_send_batch, **kwargs)


class ResendListEmails(Tool):
    name: str = "resend_list_emails"
    description: str | None = "List emails sent by your team. Use after cursor from the last item for pagination."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_emails(
            after: str | None = Field(None, description="Pagination cursor: ID of the last email from the previous page"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            params: dict[str, str] = {}
            if after:
                params["after"] = after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails",
                    headers=_headers(key),
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_emails, **kwargs)


class ResendGetEmail(Tool):
    name: str = "resend_get_email"
    description: str | None = "Retrieve a single sent email by ID, including HTML and plain text content."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_email(
            email_id: str = Field(..., description="Sent email ID"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/{email_id}",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_email, **kwargs)


class ResendUpdateEmail(Tool):
    name: str = "resend_update_email"
    description: str | None = "Update a scheduled email (e.g. reschedule via scheduled_at)."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _update_email(
            email_id: str = Field(..., description="Scheduled email ID"),
            scheduled_at: str = Field(
                ...,
                description="New scheduled send time in ISO 8601 format (e.g. 2026-08-05T11:52:01.858Z)",
            ),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_BASE_URL}/emails/{email_id}",
                    headers=_headers(key),
                    json={"scheduled_at": scheduled_at},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_email, **kwargs)


class ResendCancelEmail(Tool):
    name: str = "resend_cancel_email"
    description: str | None = "Cancel a scheduled email before it is sent."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_email(
            email_id: str = Field(..., description="Scheduled email ID to cancel"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/emails/{email_id}/cancel",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_cancel_email, **kwargs)


class ResendListEmailAttachments(Tool):
    name: str = "resend_list_email_attachments"
    description: str | None = "List attachments for a sent email."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_attachments(
            email_id: str = Field(..., description="Sent email ID"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/{email_id}/attachments",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_attachments, **kwargs)


class ResendGetEmailAttachment(Tool):
    name: str = "resend_get_email_attachment"
    description: str | None = "Retrieve metadata and download URL for a sent email attachment."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_attachment(
            email_id: str = Field(..., description="Sent email ID"),
            attachment_id: str = Field(..., description="Attachment ID"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/{email_id}/attachments/{attachment_id}",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_attachment, **kwargs)


class ResendListReceivedEmails(Tool):
    name: str = "resend_list_received_emails"
    description: str | None = "List inbound emails received by your domain. Use after cursor for pagination."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_received(
            after: str | None = Field(None, description="Pagination cursor: ID of the last email from the previous page"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            params: dict[str, str] = {}
            if after:
                params["after"] = after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/receiving",
                    headers=_headers(key),
                    params=params or None,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_received, **kwargs)


class ResendGetReceivedEmail(Tool):
    name: str = "resend_get_received_email"
    description: str | None = "Retrieve a single inbound email by ID, including body and headers."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_received(
            email_id: str = Field(..., description="Received email ID"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/receiving/{email_id}",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_received, **kwargs)


class ResendListReceivedAttachments(Tool):
    name: str = "resend_list_received_attachments"
    description: str | None = "List attachments for an inbound received email."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_received_attachments(
            email_id: str = Field(..., description="Received email ID"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/receiving/{email_id}/attachments",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_received_attachments, **kwargs)


class ResendGetReceivedAttachment(Tool):
    name: str = "resend_get_received_attachment"
    description: str | None = "Retrieve metadata and download URL for an inbound email attachment."
    integration: Annotated[str, Integration("resend")] | None = None
    api_key: SecretStr | None = None
    default_from: str | None = None
    default_reply_to: list[str] | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_received_attachment(
            email_id: str = Field(..., description="Received email ID"),
            attachment_id: str = Field(..., description="Attachment ID"),
        ) -> dict:
            key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/emails/receiving/{email_id}/attachments/{attachment_id}",
                    headers=_headers(key),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_received_attachment, **kwargs)
