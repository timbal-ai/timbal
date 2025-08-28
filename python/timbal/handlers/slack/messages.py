"""
Slack Messages Module

Setup:
1. Create app at https://api.slack.com/apps
2. Configure permissions.
    In order to retrieve messages, you need to allow the following permissions depending on the type of action you want to perform.
    For example:
    - chat:write (for sending messages to a channel you're a member of)
    - conversations:history (for retrieving messages from a public channel)
3. Install app to workspace
4. Set environment variables:
   - SLACK_BOT_TOKEN: the message will be sent by the bot
   - SLACK_USER_TOKEN: the message will be sent by the user

Schedule messages are not supported by the user token.

Example Usage:
>>> get_messages(channel="general")
>>> send_message(channel="general", text="Hello, world!")
>>> schedule_message(channel="general", text="Hello, world!", minutes_from_now=1)
"""
import os
from datetime import datetime, timedelta

from pydantic import Field
from slack_sdk import WebClient

from ...errors import APIKeyNotFoundError
from ...utils import resolve_default


def get_messages(
    channel: str = Field(
        ...,
        description="The channel id or channel name to get the messages for."
    ),
    cursor: str | None = Field(
        None,
        description="Pagination cursor."
        "Leave blank to start from the beginning of the messages."
    ),
    include_all_metadata: bool = Field(
        False,
        description="Include all metadata in the messages. Defaults to false"
    ),
    inclusive: bool = Field(
        False,
        description="Include messages with oldest or latest timestamps in results."
        "Ignored unless either timestamp is specified."
    ),
    limit: int = Field(
        100,
        description="The maximum number of items to return. "
        "Fewer than the requested number of items may be returned, even if the end of "
        "the conversation history hasn't been reached. Maximum of 999. Defaults to 100"
    ),
    latest: str | None = Field(
        None,
        description="Only messages before this Unix timestamp will be included in results. "
        "Default is the current time."
    ),
    oldest: str | None = Field(
        None,
        description="Only messages after this Unix timestamp will be included in results. "
        "Default is the current time."
    )
) -> dict:
    """This function can be used to get messages from a specific channel."""
    cursor = resolve_default("cursor", cursor)
    include_all_metadata = resolve_default("include_all_metadata", include_all_metadata)
    inclusive = resolve_default("inclusive", inclusive)
    limit = resolve_default("limit", limit)
    latest = resolve_default("latest", latest)
    oldest = resolve_default("oldest", oldest)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")
        
    client = WebClient(token=token)
    result = client.conversations_history(
        channel=channel,
        cursor=cursor,
        include_all_metadata=include_all_metadata,
        inclusive=inclusive,
        limit=limit,
        latest=latest,
        oldest=oldest
    )
    messages = result["messages"]
    return messages


def get_replies(
    channel: str = Field(
        ...,
        description="The channel id or channel name to get the messages for."
    ),
    ts: str = Field(
        ...,
        description="The timestamp of the message to get the replies for."
    ),
    # TODO Add more stuff.
) -> dict:

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")

    client = WebClient(token=token)
    result = client.conversations_replies(
        channel=channel,
        ts=ts,
    )
    return result


def send_message(
    channel: str = Field(
        ...,
        description="The channel id or channel name to send message to."
        "If you want to send a message to a user, you can use the user's id."
    ),
    text: str | None = Field(
        None,
        description="The message to be sent"
    ),
    attachments: str | None = Field(
        None,
        description="A JSON-based array of structured attachments, presented as a URL-encoded string."
    ),
    blocks: str | None = Field(
        None,
        description="A JSON-based array of structured blocks, presented as a URL-encoded string."
    ),
    icon_emoji: str | None = Field(
        None,
        description="Emoji to use as the icon for this message. Overrides icon_url."
    ),
    icon_url: str | None = Field(
        None,
        description="URL to an image to use as the icon for this message."
    ),
    link_names: bool = Field(
        False,
        description="Find and link user groups."
        "No longer supports linking individual users; use syntax shown in Mentioning Users instead."
    ),
    markdown_text: str | None = Field(
        None,
        description="Accepts message text formatted in markdown. "
        "This argument should not be used in conjunction with blocks or text. "
        "Limit this field to 12,000 characters."
    ),
    metadata: str | None = Field(
        None,
        description="JSON object with event_type and event_payload fields, presented as a URL-encoded string."
    ),
    parse: str | None = Field(
        None,
        description="Change how messages are treated. "
        "Options are none or full. "
    ),
    reply_broadcast: bool = Field(
        False,
        description="Used in conjunction with thread_ts and indicates whether reply should be made visible to everyone in the channel or conversation. "
        "Defaults to false."
    ),
    thread_ts: str | None = Field(
        None,
        description="Provide another message's ts value to make this message a reply. "
        "Avoid using a reply's ts value; use its parent instead."
    ),
    unfurl_links: bool = Field(
        False,
        description="Pass true to enable unfurling of primarily text-based content. Default is false."
    ),
    unfurl_media: bool = Field(
        False,
        description="Pass true to enable unfurling of media content. Default is false."
    )
) -> dict:
    """This function can be used to send a message to a specific channel."""
    text = resolve_default("text", text)
    attachments = resolve_default("attachments", attachments)
    blocks = resolve_default("blocks", blocks)
    icon_emoji = resolve_default("icon_emoji", icon_emoji)
    icon_url = resolve_default("icon_url", icon_url)
    link_names = resolve_default("link_names", link_names)
    markdown_text = resolve_default("markdown_text", markdown_text)
    metadata = resolve_default("metadata", metadata)
    parse = resolve_default("parse", parse)
    reply_broadcast = resolve_default("reply_broadcast", reply_broadcast)
    thread_ts = resolve_default("thread_ts", thread_ts)
    unfurl_links = resolve_default("unfurl_links", unfurl_links)
    unfurl_media = resolve_default("unfurl_media", unfurl_media)

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")
        
    if text is None and attachments is None and blocks is None and markdown_text is None:
        raise ValueError("Either text, attachments, blocks, or markdown_text must be provided to send a message")
    
    if markdown_text is not None and (text is not None or blocks is not None):
        raise ValueError("markdown_text should not be used in conjunction with text or blocks")

    client = WebClient(token=token)

    message_params = {
        "channel": channel,
        "icon_emoji": icon_emoji,
        "icon_url": icon_url,
        "link_names": link_names,
        "metadata": metadata,
        "parse": parse,
        "reply_broadcast": reply_broadcast,
        "thread_ts": thread_ts,
        "unfurl_links": unfurl_links,
        "unfurl_media": unfurl_media
    }
    
    if text:
        message_params["text"] = text
    if attachments:
        message_params["attachments"] = attachments
    if blocks:
        message_params["blocks"] = blocks
    if markdown_text:
        message_params["text"] = markdown_text
        message_params["mrkdwn"] = True
    
    result = client.chat_postMessage(**message_params)
    return result
    

def schedule_message(
    channel: str = Field(
        ...,
        description="The channel name or ID to send message to"
    ),
    text: str | None = Field(
        None,
        description="The message to be sent"
    ),
    attachments: str | None = Field(
        None,
        description="A JSON-based array of structured attachments, presented as a URL-encoded string."
    ),
    blocks: str | None = Field(
        None,
        description="A JSON-based array of structured blocks, presented as a URL-encoded string."
    ),
    minutes_from_now: int | None = Field(
        None,
        description="Number of minutes from now to schedule the message"
    ),
    timestamp: str | None = Field(
        None,
        description="Unix timestamp representing the future time the message should post to Slack."
        "The datetime in format: YYYY-MM-DDTHH:MM:SS±hh:mm."
    ),
    link_names: bool = Field(
        False,
        description="Find and link user groups."
        "No longer supports linking individual users; use syntax shown in Mentioning Users instead."
    ),
    markdown_text: str | None = Field(
        None,
        description="Accepts message text formatted in markdown. "
        "This argument should not be used in conjunction with blocks or text. "
        "Limit this field to 12,000 characters."
    ),
    metadata: str | None = Field(
        None,
        description="JSON object with event_type and event_payload fields, presented as a URL-encoded string."
    ),
    parse: enumerate | None = Field(
        None,
        description="Change how messages are treated. "
        "Options are none or full. "
    ),
    reply_broadcast: bool = Field(
        False,
        description="Used in conjunction with thread_ts and indicates whether reply should be made visible to everyone in the channel or conversation. "
        "Defaults to false."
    ),
    thread_ts: str | None = Field(
        None,
        description="Provide another message's ts value to make this message a reply. "
        "Avoid using a reply's ts value; use its parent instead."
    ),
    unfurl_links: bool = Field(
        False,
        description="Pass true to enable unfurling of primarily text-based content. Default is false."
    ),
    unfurl_media: bool = Field(
        False,
        description="Pass true to enable unfurling of media content. Default is false."
    )
) -> dict:
    """This function can be used to schedule a message to be sent in a specific time or in a certain number of minutes from now
    It is only supported by the bot token."""
    text = resolve_default("text", text)
    attachments = resolve_default("attachments", attachments)
    blocks = resolve_default("blocks", blocks)
    minutes_from_now = resolve_default("minutes_from_now", minutes_from_now)
    timestamp = resolve_default("timestamp", timestamp)
    link_names = resolve_default("link_names", link_names)
    markdown_text = resolve_default("markdown_text", markdown_text)
    metadata = resolve_default("metadata", metadata)
    parse = resolve_default("parse", parse)
    reply_broadcast = resolve_default("reply_broadcast", reply_broadcast)
    thread_ts = resolve_default("thread_ts", thread_ts)
    unfurl_links = resolve_default("unfurl_links", unfurl_links)
    unfurl_media = resolve_default("unfurl_media", unfurl_media)

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")
    
    client = WebClient(token=token)
    
    if minutes_from_now:
        scheduled_time = datetime.now() + timedelta(minutes=minutes_from_now)
        unix_timestamp = scheduled_time.timestamp()
    elif timestamp:
        unix_timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    else:
        raise ValueError("Either minutes_from_now or timestamp must be provided to schedule a message")
    
    if text is None and attachments is None and blocks is None and markdown_text is None:
        raise ValueError("Either text, attachments, blocks, or markdown_text must be provided to schedule a message")
    
    if markdown_text is not None and (text is not None or blocks is not None):
        raise ValueError("markdown_text should not be used in conjunction with text or blocks")
    
    message_params = {
        "channel": channel,
        "post_at": int(unix_timestamp),
        "link_names": link_names,
        "metadata": metadata,
        "parse": parse,
        "reply_broadcast": reply_broadcast,
        "thread_ts": thread_ts,
        "unfurl_links": unfurl_links,
        "unfurl_media": unfurl_media
    }
    
    if text:
        message_params["text"] = text
    if attachments:
        message_params["attachments"] = attachments
    if blocks:
        message_params["blocks"] = blocks
    if markdown_text:
        message_params["text"] = markdown_text
        message_params["mrkdwn"] = True
    
    result = client.chat_scheduleMessage(**message_params)
    
    return result


def delete_message(
    channel: str = Field(
        ...,
        description="The channel id or channel name to delete the message from."
    ),
    ts: str = Field(
        ...,
        description="The timestamp of the message to delete."
    ),
) -> dict:

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")
    
    client = WebClient(token=token)
    result = client.chat_delete(
        channel=channel,
        ts=ts,
    )
    return result


def edit_message(
    channel: str = Field(
        ...,
        description="The channel id or channel name to edit the message from."
    ),
    ts: str = Field(
        ...,
        description="The timestamp of the message to edit."
    ),
    text: str | None = Field(
        None,
        description="The new text of the message."
    ),
) -> dict:
    text = resolve_default("text", text)

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")
    
    client = WebClient(token=token)
    result = client.chat_update(
        channel=channel,
        ts=ts,
        text=text,
    )
    return result
