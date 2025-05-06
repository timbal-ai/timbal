"""
Gmail Message Operations

This module provides a comprehensive set of functions to interact with Gmail messages using the Gmail API.

Features:
- Create drafts and send emails
- Retrieve messages from inbox
- Search through emails
- Handle attachments

Prerequisites:
- Gmail API authentication must be completed (see authenticate.py)
- Valid credentials and token files must be present

Example Usage:
>>> send_email(to="recipient@example.com", subject="Hello", body="This is a test email")
>>> get_messages(query="from:someone@example.com")
>>> create_draft(to="recipient@example.com", subject="Draft", body="Draft content")

Note: All operations require proper Gmail API authentication and appropriate scopes.
Make sure you have the necessary permissions configured in your Google Cloud Console.
"""
import base64
import mimetypes
import os
import os.path
from email.message import EmailMessage
from typing import Any, Literal

from ...types import Field
from .authenticate import get_gmail_client


def create_draft_message(
    to: list[str] = Field(
        description="The email addresses of the recipients"
    ),
    subject: str | None = Field(
        default=None,
        description="The subject of the email"
    ),
    body: str | None = Field(
        default=None,
        description="The body of the email"
    ),   
    from_email: str | None = Field(
        default=None,
        description="The email address of the sender"
    ),
    cc: list[str] | None = Field(
        default=None,
        description="The email addresses of the recipients to be included in the 'Cc' field"
    ), 
    bcc: list[str] | None = Field(
        default=None,
        description="The email addresses of the recipients to be included in the 'Bcc' field"
    ), 
    attachment: str | None = Field(
        default=None,
        description="The path to the attachment file"
    )
) -> dict:
    """This function can be used to create a draft email message."""

    # Enable calling this step without pydantic model_validate()
    subject = subject.default if hasattr(subject, 'default') else subject
    body = body.default if hasattr(body, 'default') else body
    to = to.default if hasattr(to, 'default') else to
    from_email = from_email.default if hasattr(from_email, 'default') else from_email
    cc = cc.default if hasattr(cc, 'default') else cc
    bcc = bcc.default if hasattr(bcc, 'default') else bcc
    attachment = attachment.default if hasattr(attachment, 'default') else attachment

    service = get_gmail_client()
    draft_message = EmailMessage()

    try:
        draft_message.set_content(body)
        draft_message["To"] = ", ".join(to)
        draft_message["Subject"] = subject
        if from_email is not None:
            draft_message["From"] = from_email
        if cc is not None:
            draft_message["Cc"] = ", ".join(cc)

        if bcc is not None:
            draft_message["Bcc"] = ", ".join(bcc)

        if attachment is not None:
            attachment_filename = attachment
            type_subtype, _ = mimetypes.guess_type(attachment_filename)
            maintype, subtype = type_subtype.split("/")
            with open(attachment_filename, "rb") as fp:
                attachment_data = fp.read()
            file_name = os.path.basename(attachment)
            draft_message.add_attachment(attachment_data, maintype, subtype, filename=file_name)

        encoded_message = base64.urlsafe_b64encode(draft_message.as_bytes()).decode()
        create_draft_request_body = {"message": {"raw": encoded_message}}
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body=create_draft_request_body)
            .execute()
        )

        return {
            'id': draft["id"],
            'subject': subject,
            'body': body,
            'to': ", ".join(to),
            'from': from_email,
            'cc': ", ".join(cc) if cc else None,
            'bcc': ", ".join(bcc) if bcc else None,
            'attachment': os.path.basename(attachment) if attachment else None
        }
    except Exception as e:
        raise ValueError(f"Error setting content: {e}") from e



def get_message(
    message_id: str = Field(
        description="The ID of the message to get"
    ),
    message_format: str = Field(
        default="full",
        description="The format of the message to get. ['minimal', 'full', 'raw', 'metadata']"
    )
) -> dict:
    """This function can be used to get a message from a specific email address."""
    # Enable calling this step without pydantic model_validate()
    message_format = message_format.default if hasattr(message_format, "default") else message_format

    service = get_gmail_client()
    try:
        query = (
            service.users()
            .messages()
            .get(userId="me", format=message_format, id=message_id)
        )
        message_data = query.execute()

        return message_data
    except Exception as e:
        raise ValueError(f"Error getting message: {e}") from e



def get_thread(
    thread_id: str = Field(
        description="The ID of the thread to get"
    ),
    message_format: Literal["minimal", "full", "raw", "metadata"] = Field(
        default="full",
        description="The format of the message to get. ['minimal', 'full', 'raw', 'metadata']"
    )
) -> dict:
    """Retrieve a Gmail thread and its messages.

    This function fetches a complete email thread (conversation) from Gmail, including all messages
    in the thread. The thread ID can be obtained from the search function or from a message's
    threadId field.

    Args:
        thread_id: The unique identifier of the thread to retrieve
        message_format: The format of the returned messages.

    Returns:
        dict: A dictionary containing the thread information.

    Raises:
        ValueError: If there's an error retrieving the thread or if the thread ID is invalid
    """
    # Enable calling this step without pydantic model_validate()
    message_format = message_format.default if hasattr(message_format, "default") else message_format

    service = get_gmail_client()

    try:
        query = (
            service.users()
            .threads()
            .get(userId="me", id=thread_id, format=message_format)
        )
        thread_data = query.execute()
        
    except Exception as e:
        raise ValueError(f"Error getting thread: {e}") from e
    
    return thread_data



def send_message(
    to: list[str] = Field(
        description="The email addresses of the recipients"
    ), 
    body: str | None = Field(
        default=None,
        description="The body of the email"
    ),
    subject: str | None = Field(
        default=None,
        description="The subject of the email"
    ), 
    cc: list[str] | None = Field(
        default=None,
        description="The email addresses of the recipients to be included in the 'Cc' field"
    ), 
    bcc: list[str] | None = Field(
        default=None,
        description="The email addresses of the recipients to be included in the 'Bcc' field"
    ),
    attachment: str | None = Field(
        description="The path to the attachment file"
    )
) -> dict:
    """Send an email using Gmail API.

    This function allows you to send emails with optional CC, BCC recipients and attachments.
    The email will be sent from the authenticated Gmail account.

    Args:
        to: List of recipient email addresses
        body: The main content of the email
        subject: The subject line of the email
        cc: Optional list of CC recipient email addresses
        bcc: Optional list of BCC recipient email addresses
        attachment: Optional path to a file to attach

    Returns:
        dict: A dictionary containing information about the sent email:

    Raises:
        ValueError: If there's an error creating or sending the email
    """
    # Enable calling this step without pydantic model_validate()
    subject = subject.default if hasattr(subject, 'default') else subject
    body = body.default if hasattr(body, 'default') else body
    cc = cc.default if hasattr(cc, 'default') else cc
    bcc = bcc.default if hasattr(bcc, 'default') else bcc
    attachment = attachment.default if hasattr(attachment, 'default') else attachment

    service = get_gmail_client()
    message = EmailMessage()

    try:
        message.set_content(body)
        message["To"] = ", ".join(to if isinstance(to, list) else [to])
        message["Subject"] = subject

        if cc:
            message["Cc"] = ", ".join(cc if isinstance(cc, list) else [cc])

        if bcc:
            message["Bcc"] = ", ".join(bcc if isinstance(bcc, list) else [bcc])

        if attachment is not None:
            attachment_filename = attachment
            type_subtype, _ = mimetypes.guess_type(attachment_filename)
            maintype, subtype = type_subtype.split("/")
            with open(attachment_filename, "rb") as fp:
                attachment_data = fp.read()
            file_name = os.path.basename(attachment)
            message.add_attachment(attachment_data, maintype, subtype, filename=file_name)

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )

        return {
            'id': send_message["id"],
            'subject': subject,
            'body': body,
            'to': ", ".join(to),
            'cc': ", ".join(cc) if cc else None,
            'bcc': ", ".join(bcc) if bcc else None,
            'attachment': os.path.basename(attachment) if attachment else None
        }

    except Exception as e:
        raise ValueError(f"Error setting content: {e}") from e



def search(
    query: str = Field(
        description="The search query using Gmail's search syntax"
    ), 
    resource: Literal["messages", "threads"] = Field(
        default="messages",
        description="The type of resource to search: 'messages' or 'threads'"
    ), 
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return"
    )
) -> list[dict[str, Any]]:
    """
    Search Gmail messages or threads using Gmail's search syntax.
    
    Args:
        query: Search query using Gmail's search syntax
        resource: Type of resource to search ("messages" or "threads")
        max_results: Maximum number of results to return
        
    Returns:
        List of message or thread objects matching the search criteria
        
    Raises:
        ValueError: If resource type is not supported
    """
    # Enable calling this step without pydantic model_validate()
    max_results = max_results.default if hasattr(max_results, "default") else max_results
    resource = resource.default if hasattr(resource, "default") else resource

    service = get_gmail_client()
    
    if resource == "messages":
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
            .get(resource, [])
        )
        return results
    elif resource == "threads":
        results = (
            service.users()
            .threads()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
            .get(resource, [])
        )
        return results
    else:
        raise ValueError(f"Resource type '{resource}' not supported. Use 'messages' or 'threads'.")