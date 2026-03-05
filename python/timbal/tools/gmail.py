import base64
import mimetypes
import re
from datetime import datetime
from html import escape as html_escape
from typing import Annotated, Any, Literal

from pydantic import Field

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


async def _download_and_encode_attachment(url: str) -> tuple[str, str, str]:
    """Download file from URL and return filename, content_type, and base64 content."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            filename = url.split("/")[-1] if "/" in url else "attachment"
            if "Content-Disposition" in response.headers:
                content_disposition = response.headers["Content-Disposition"]
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')

            content_type = response.headers.get("content-type", "application/octet-stream")
            if not content_type or content_type == "application/octet-stream":
                detected_type, _ = mimetypes.guess_type(filename)
                if detected_type:
                    content_type = detected_type

            file_content = response.content
            base64_content = base64.b64encode(file_content).decode()

            return filename, content_type, base64_content

    except httpx.HTTPError as e:
        raise ValueError(f"Failed to download attachment from {url}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error processing attachment from {url}: {e}") from e


def _extract_message_body(payload: dict) -> tuple[str, str]:
    """Extract HTML and text body from Gmail message payload."""
    body_html = ""
    body_text = ""

    def extract_parts(part: dict):
        nonlocal body_html, body_text

        if part.get("mimeType") == "text/html" and "data" in part.get("body", {}):
            body_html = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
        elif part.get("mimeType") == "text/plain" and "data" in part.get("body", {}):
            body_text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
        elif "parts" in part:
            for subpart in part["parts"]:
                extract_parts(subpart)

    extract_parts(payload)
    return body_html, body_text


def _build_quoted_html(
    from_email: str,
    date: str,
    body_html: str,
    body_text: str,
    is_forward: bool = False,
    subject: str = "",
    to_email: str = "",
) -> str:
    """Build Gmail-style quoted message HTML."""
    original_content = body_html
    if not original_content and body_text:
        original_content = html_escape(body_text).replace("\n", "<br>")

    if not original_content:
        return ""

    try:
        parsed_date = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = parsed_date.strftime("%a, %d %b %Y %H:%M")
    except (ValueError, TypeError):
        formatted_date = date

    if is_forward:
        quoted_original_html = (
            "<br>\n"
            '<div class="gmail_quote">'
            '<div dir="ltr" class="gmail_attr">'
            "---------- Forwarded message ---------<br>"
            f"From: <b>{html_escape(from_email)}</b><br>"
            f"Date: {html_escape(formatted_date)}<br>"
            f"Subject: {html_escape(subject)}<br>"
            f"To: {html_escape(to_email)}<br>"
            "</div>"
            "<br>"
            f"{original_content}"
            "</div>"
        )
    else:
        attribution = f"On {html_escape(formatted_date)}, &lt;{html_escape(from_email)}&gt; wrote:<br>"
        quoted_original_html = (
            "<br>\n"
            '<div class="gmail_quote">'
            f'<div dir="ltr" class="gmail_attr">{attribution}<br></div>'
            '<blockquote class="gmail_quote" style="margin:0px 0px 0px 0.8ex;'
            'border-left:1px solid rgb(204,204,204);padding-left:1ex">'
            f"{original_content}"
            "</blockquote>"
            "</div>"
        )

    return quoted_original_html


def _build_quoted_text(
    from_email: str,
    date: str,
    body_html: str,
    body_text: str,
) -> str:
    """Build plain-text quoted reply from the original message."""
    original_content = body_text or body_html
    if not original_content:
        return ""

    try:
        parsed_date = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = parsed_date.strftime("%a, %d %b %Y %H:%M")
    except (ValueError, TypeError):
        formatted_date = date

    if body_html and not body_text:
        original_content = re.sub(r"<[^>]+>", "", body_html)

    return f"\n\nOn {formatted_date}, <{from_email}> wrote:\n> {original_content.replace(chr(10), chr(10) + '> ')}"


class GmailSend(Tool):
    name: str = "gmail_send"
    description: str | None = "Send an email via Gmail."
    integration: Annotated[str, Integration("gmail")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        config = {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }
        return config

    def __init__(self, **kwargs: Any) -> None:
        async def _send_email(
            to: str | list[str] = Field(..., description="Recipient email address(es)"),
            subject: str = Field(..., description="Email subject"),
            body: str = Field(..., description="Email body content"),
            cc: str | list[str] | None = Field(None, description="CC recipient(s)"),
            bcc: str | list[str] | None = Field(None, description="BCC recipient(s)"),
            attachment_urls: list[str] | None = Field(None, description="URLs of files to attach"),
            message_format: Literal["text", "html"] = Field("text", description='"text" or "html"'),
            email_signature: str | None = Field(None, description="Email signature to append"),
        ) -> Any:
            if not isinstance(self.integration, Integration):
                raise ValueError("Gmail integration not configured.")
            credentials = await self.integration.resolve()
            api_key = credentials["token"]
            import httpx

            if isinstance(to, list):
                to_str = ", ".join(to)
            else:
                to_str = to

            message_parts = [f"To: {to_str}"]

            if cc:
                cc_str = ", ".join(cc) if isinstance(cc, list) else cc
                message_parts.append(f"Cc: {cc_str}")

            if bcc:
                bcc_str = ", ".join(bcc) if isinstance(bcc, list) else bcc
                message_parts.append(f"Bcc: {bcc_str}")

            message_parts.append(f"Subject: {subject}")

            if attachment_urls or message_format == "html" or email_signature:
                boundary = "----=_NextPart_" + base64.urlsafe_b64encode(str(hash(body)).encode()).decode()[:16]
                message_parts.append("MIME-Version: 1.0")
                message_parts.append(f'Content-Type: multipart/mixed; boundary="{boundary}"')
                message_parts.append("")

                message_parts.append(f"--{boundary}")
                if message_format == "html":
                    message_parts.append("Content-Type: text/html; charset=utf-8")
                    message_parts.append("Content-Transfer-Encoding: 7bit")
                    message_parts.append("")

                    signature_html = ""
                    if email_signature:
                        signature_parts = ["<br><br>", "<!-- Signature -->"]
                        if not email_signature.strip().startswith("<div"):
                            signature_parts.extend(
                                [
                                    '<div style="font-size:13px; color:#444; margin-top: 20px;">',
                                    f"{email_signature}",
                                    "</div>",
                                ]
                            )
                        else:
                            signature_parts.append(f"{email_signature}")
                        signature_html = "\n".join(signature_parts)

                    body_content = body + signature_html
                    message_parts.append(body_content)
                else:
                    message_parts.append("Content-Type: text/plain; charset=utf-8")
                    message_parts.append("Content-Transfer-Encoding: 7bit")
                    message_parts.append("")
                    body_content = body
                    if email_signature:
                        body_content += "\n\n" + email_signature
                    message_parts.append(body_content)
                message_parts.append("")

                if attachment_urls:
                    for url in attachment_urls:
                        try:
                            filename, att_content_type, base64_content = await _download_and_encode_attachment(url)

                            message_parts.append(f"--{boundary}")
                            message_parts.append(f"Content-Type: {att_content_type}")
                            message_parts.append("Content-Transfer-Encoding: base64")
                            message_parts.append(f'Content-Disposition: attachment; filename="{filename}"')
                            message_parts.append("")
                            message_parts.append(base64_content)
                            message_parts.append("")
                        except ValueError as e:
                            raise ValueError(f"Failed to attach {url}: {e}") from e

                message_parts.append(f"--{boundary}--")
            else:
                content_type = "text/html" if message_format == "html" else "text/plain"
                message_parts.append(f"Content-Type: {content_type}; charset=utf-8")
                message_parts.append("")
                if email_signature:
                    message_parts.append(body + "\n\n" + email_signature)
                else:
                    message_parts.append(body)

            message = "\r\n".join(message_parts)
            raw = base64.urlsafe_b64encode(message.encode()).decode()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/messages/send",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"raw": raw},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_send_email, **kwargs)


class GmailReply(Tool):
    name: str = "gmail_reply"
    description: str | None = "Reply to an email thread in Gmail."
    integration: Annotated[str, Integration("gmail")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        config = {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }
        return config

    def __init__(self, **kwargs: Any) -> None:
        async def _reply_to_email(
            email_conversation: str = Field(..., description="Message ID to reply to"),
            reply_message: str = Field(..., description="Reply body content"),
            reply_all: bool = Field(False, description="Reply to all recipients"),
            reply_to_specific_message: bool = Field(False, description="Reply to specific message instead of thread"),
            override_recipients: list[str] | None = Field(None, description="Override default recipients"),
            content_type: Literal["text", "html"] = Field("text", description='"text" or "html"'),
            email_signature: str | None = Field(None, description="Email signature to append"),
            attachments: list[str] | None = Field(None, description="URLs of files to attach"),
        ) -> Any:
            if not isinstance(self.integration, Integration):
                raise ValueError("Gmail integration not configured.")
            credentials = await self.integration.resolve()
            api_key = credentials["token"]
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/messages/{email_conversation}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={
                        "format": "full",
                        "metadataHeaders": ["Subject", "From", "To", "Cc", "References", "Message-ID", "Thread-Id"],
                    },
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                original_message = response.json()

            headers = {header["name"].upper(): header["value"] for header in original_message["payload"]["headers"]}
            subject = headers.get("SUBJECT", "")
            from_email = headers.get("FROM", "")
            to_email = headers.get("TO", "")
            cc_email = headers.get("CC", "")
            message_id_header = headers.get("MESSAGE-ID", "")
            references_header = headers.get("REFERENCES", "")
            thread_id = original_message["threadId"]
            date_header = headers.get("DATE", "")

            original_body_html, original_body_text = _extract_message_body(original_message["payload"])

            if not subject.startswith("Re:"):
                reply_subject = f"Re: {subject}"
            else:
                reply_subject = subject

            to_recipients = []
            cc_recipients = []

            if override_recipients:
                to_recipients = override_recipients
            elif reply_all:
                to_recipients.append(from_email.split("<")[-1].replace(">", "").strip())
                if to_email:
                    to_recipients.extend([email.strip() for email in to_email.split(",")])
                if cc_email:
                    cc_recipients.extend([email.strip() for email in cc_email.split(",")])
            else:
                to_recipients.append(from_email.split("<")[-1].replace(">", "").strip())

            message_parts = []
            message_parts.append(f"To: {', '.join(to_recipients)}")

            if cc_recipients:
                message_parts.append(f"Cc: {', '.join(cc_recipients)}")

            message_parts.append(f"Subject: {reply_subject}")

            if message_id_header:
                message_parts.append(f"In-Reply-To: {message_id_header}")
                if not reply_to_specific_message and references_header:
                    message_parts.append(f"References: {references_header} {message_id_header}")
                else:
                    message_parts.append(f"References: {message_id_header}")

            if attachments or content_type == "html" or email_signature:
                boundary = "----=_NextPart_" + base64.urlsafe_b64encode(str(hash(reply_message)).encode()).decode()[:16]
                message_parts.append("MIME-Version: 1.0")
                message_parts.append(f'Content-Type: multipart/mixed; boundary="{boundary}"')
                message_parts.append("")

                message_parts.append(f"--{boundary}")
                if content_type == "html":
                    message_parts.append("Content-Type: text/html; charset=utf-8")

                    signature_html = ""
                    if email_signature:
                        signature_parts = ["<br><br>", "<!-- Signature -->"]
                        if not email_signature.strip().startswith("<div"):
                            signature_parts.extend(
                                [
                                    '<div style="font-size:13px; color:#444; margin-top: 20px;">',
                                    f"{email_signature}",
                                    "</div>",
                                ]
                            )
                        else:
                            signature_parts.append(f"{email_signature}")
                        signature_html = "\n".join(signature_parts)

                    quoted_original_html = _build_quoted_html(
                        from_email=from_email,
                        date=date_header,
                        body_html=original_body_html,
                        body_text=original_body_text,
                        is_forward=False,
                    )

                    body_html = reply_message
                    if signature_html:
                        body_html += signature_html
                    if quoted_original_html:
                        body_html += quoted_original_html

                    body_content = "\n".join(
                        [
                            "<!DOCTYPE html>",
                            "<html>",
                            "  <head>",
                            '    <meta charset="UTF-8">',
                            "  </head>",
                            '  <body style="margin:0; padding:0; font-family: Arial, sans-serif;">',
                            f"    {body_html}",
                            "  </body>",
                            "</html>",
                        ]
                    )
                else:
                    message_parts.append("Content-Type: text/plain; charset=utf-8")
                    body_content = reply_message
                    if email_signature:
                        body_content += f"\n\n{email_signature}"
                    body_content += _build_quoted_text(from_email, date_header, original_body_html, original_body_text)

                message_parts.append("Content-Transfer-Encoding: 7bit")
                message_parts.append("")
                message_parts.append(body_content)
                message_parts.append("")

                if attachments:
                    for url in attachments:
                        try:
                            filename, att_content_type, base64_content = await _download_and_encode_attachment(url)

                            message_parts.append(f"--{boundary}")
                            message_parts.append(f"Content-Type: {att_content_type}")
                            message_parts.append("Content-Transfer-Encoding: base64")
                            message_parts.append(f'Content-Disposition: attachment; filename="{filename}"')
                            message_parts.append("")
                            message_parts.append(base64_content)
                            message_parts.append("")
                        except ValueError as e:
                            raise ValueError(f"Warning: Failed to attach {url}: {e}") from e

                message_parts.append(f"--{boundary}--")
            else:
                message_parts.append("Content-Type: text/plain; charset=utf-8")
                message_parts.append("")
                body_content = reply_message
                if email_signature:
                    body_content += f"\n\n{email_signature}"
                body_content += _build_quoted_text(from_email, date_header, original_body_html, original_body_text)
                message_parts.append(body_content)

            message = "\r\n".join(message_parts)
            raw = base64.urlsafe_b64encode(message.encode()).decode()

            async with httpx.AsyncClient() as client:
                send_data = {"raw": raw}
                if not reply_to_specific_message:
                    send_data["threadId"] = thread_id

                response = await client.post(
                    f"{_BASE_URL}/messages/send",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=send_data,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reply_to_email, **kwargs)


class GmailSearch(Tool):
    name: str = "gmail_search"
    description: str | None = "Search for emails in Gmail."
    integration: Annotated[str, Integration("gmail")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        config = {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }
        return config

    def __init__(self, **kwargs: Any) -> None:
        async def _search_emails(
            search_in_gmail: str = Field(..., description="Gmail search query"),
            max_results: int = Field(10, description="Maximum number of results"),
            include_attachments: bool = Field(False, description="Include attachment metadata"),
        ) -> Any:
            if not isinstance(self.integration, Integration):
                raise ValueError("Gmail integration not configured.")
            credentials = await self.integration.resolve()
            api_key = credentials["token"]
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/messages",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"q": search_in_gmail, "maxResults": max_results},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                search_results = response.json()

            messages = search_results.get("messages", [])
            full_messages = []

            async with httpx.AsyncClient() as detail_client:
                for msg in messages:
                    msg_id = msg["id"]
                    msg_response = await detail_client.get(
                        f"{_BASE_URL}/messages/{msg_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        params={"format": "full"},
                        timeout=httpx.Timeout(10.0, read=None),
                    )
                    msg_response.raise_for_status()
                    full_msg = msg_response.json()

                    payload = full_msg.get("payload", {})
                    headers = {header["name"].lower(): header["value"] for header in payload.get("headers", [])}

                    snippet = full_msg.get("snippet", "")
                    body = ""
                    attachments_data = []

                    if "parts" in payload:
                        for part in payload["parts"]:
                            if part["mimeType"] == "text/plain" and "data" in part["body"]:
                                body = base64.urlsafe_b64decode(part["body"]["data"]).decode()
                            elif part["mimeType"] == "text/html" and "data" in part["body"]:
                                if not body:
                                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode()
                            elif include_attachments and "attachmentId" in part["body"]:
                                attachments_data.append(
                                    {
                                        "filename": part.get("filename"),
                                        "mimeType": part.get("mimeType"),
                                        "attachmentId": part["body"]["attachmentId"],
                                    }
                                )
                    elif "data" in payload["body"]:
                        if payload["mimeType"] == "text/plain":
                            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode()
                        elif payload["mimeType"] == "text/html":
                            if not body:
                                body = base64.urlsafe_b64decode(payload["body"]["data"]).decode()

                    full_messages.append(
                        {
                            "id": full_msg["id"],
                            "threadId": full_msg["threadId"],
                            "snippet": snippet,
                            "subject": headers.get("subject"),
                            "from": headers.get("from"),
                            "to": headers.get("to"),
                            "date": headers.get("date"),
                            "body": body,
                            "attachments": attachments_data if include_attachments else [],
                        }
                    )

            return {
                "messages": full_messages,
                "totalResults": len(full_messages),
                "query": search_in_gmail,
                "maxResults": max_results,
            }

        super().__init__(handler=_search_emails, **kwargs)


class GmailAddLabel(Tool):
    name: str = "gmail_add_label"
    description: str | None = "Add labels to an email in Gmail."
    integration: Annotated[str, Integration("gmail")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        config = {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }
        return config

    def __init__(self, **kwargs: Any) -> None:
        async def _add_label_to_email(
            email_to_label: str = Field(..., description="Message ID to label"),
            labels: list[str] = Field(..., description="Label names to add"),
        ) -> Any:
            if not isinstance(self.integration, Integration):
                raise ValueError("Gmail integration not configured.")
            credentials = await self.integration.resolve()
            api_key = credentials["token"]
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/labels",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                labels_data = response.json()

            label_ids = []
            for label_name_item in labels:
                label_id = None
                for label in labels_data.get("labels", []):
                    if label["name"].lower() == label_name_item.lower():
                        label_id = label["id"]
                        break
                if label_id:
                    label_ids.append(label_id)
                else:
                    return {"status": "error", "message": f"Label '{label_name_item}' not found."}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/messages/{email_to_label}/modify",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"addLabelIds": label_ids},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                modified_message = response.json()

            return {
                "status": "success",
                "message": f"Labels {', '.join(labels)} added to email {email_to_label}.",
                "modified_message_id": modified_message["id"],
                "labels_added": labels,
            }

        super().__init__(handler=_add_label_to_email, **kwargs)


class GmailListLabels(Tool):
    name: str = "gmail_list_labels"
    description: str | None = "List all Gmail labels."
    integration: Annotated[str, Integration("gmail")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        config = {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }
        return config

    def __init__(self, **kwargs: Any) -> None:
        async def _list_labels(
            include_system_labels: bool = Field(True, description="Include system labels"),
            include_user_labels: bool = Field(True, description="Include user-created labels"),
        ) -> Any:
            if not isinstance(self.integration, Integration):
                raise ValueError("Gmail integration not configured.")
            credentials = await self.integration.resolve()
            api_key = credentials["token"]
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/labels",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                labels_data = response.json()

            labels = labels_data.get("labels", [])
            filtered_labels = []

            for label in labels:
                label_type = label.get("type", "")

                if label_type == "system" and not include_system_labels:
                    continue
                elif label_type == "user" and not include_user_labels:
                    continue

                filtered_labels.append(
                    {
                        "id": label["id"],
                        "name": label["name"],
                        "type": label_type,
                        "messageListVisibility": label.get("messageListVisibility", ""),
                        "labelListVisibility": label.get("labelListVisibility", ""),
                        "messagesTotal": label.get("messagesTotal", 0),
                        "messagesUnread": label.get("messagesUnread", 0),
                        "threadsTotal": label.get("threadsTotal", 0),
                        "threadsUnread": label.get("threadsUnread", 0),
                        "color": label.get("color", {}),
                    }
                )

            return {
                "labels": filtered_labels,
                "totalLabels": len(filtered_labels),
                "systemLabelsIncluded": include_system_labels,
                "userLabelsIncluded": include_user_labels,
            }

        super().__init__(handler=_list_labels, **kwargs)


class GmailRemoveLabel(Tool):
    name: str = "gmail_remove_label"
    description: str | None = "Remove labels from an email in Gmail."
    integration: Annotated[str, Integration("gmail")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        config = {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration},
                required={"integration"},
            ),
        }
        return config

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_label_from_email(
            email_to_update: str = Field(..., description="Message ID to update"),
            labels: list[str] = Field(..., description="Label names to remove"),
        ) -> Any:
            if not isinstance(self.integration, Integration):
                raise ValueError("Gmail integration not configured.")
            credentials = await self.integration.resolve()
            api_key = credentials["token"]
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/labels",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                labels_data = response.json()

            label_ids_to_remove = []
            for label_name_item in labels:
                label_id = None
                for label in labels_data.get("labels", []):
                    if label["name"].lower() == label_name_item.lower():
                        label_id = label["id"]
                        break
                if label_id:
                    label_ids_to_remove.append(label_id)
                else:
                    return {"status": "error", "message": f"Label '{label_name_item}' not found."}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/messages/{email_to_update}/modify",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"removeLabelIds": label_ids_to_remove},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                modified_message = response.json()

            return {
                "status": "success",
                "message": f"Labels {', '.join(labels)} removed from email {email_to_update}.",
                "modified_message_id": modified_message["id"],
                "labels_removed": labels,
            }

        super().__init__(handler=_remove_label_from_email, **kwargs)
