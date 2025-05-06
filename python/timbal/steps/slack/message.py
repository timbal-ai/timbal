"""
Slack Message Module

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

from slack_sdk import WebClient

from ...errors import APIKeyNotFoundError
from ...types.field import Field


def get_messages(
    channel: str = Field(
        description="The channel id or channel name to get the messages for."
    ),
    cursor: str | None = Field(
        default=None,
        description="Pagination cursor."
        "Leave blank to start from the beginning of the messages."
    ),
    include_all_metadata: bool = Field(
        default=False,
        description="Include all metadata in the messages. Defaults to false"
    ),
    inclusive: bool = Field(
        default=False,
        description="Include messages with oldest or latest timestamps in results."
        "Ignored unless either timestamp is specified."
    ),
    limit: int = Field(
        default=100,
        description="The maximum number of items to return. "
        "Fewer than the requested number of items may be returned, even if the end of "
        "the conversation history hasn't been reached. Maximum of 999. Defaults to 100"
    ),
    latest: str | None = Field(
        default=None,
        description="Only messages before this Unix timestamp will be included in results. "
        "Default is the current time."
    ),
    oldest: str | None = Field(
        default=None,
        description="Only messages after this Unix timestamp will be included in results. "
        "Default is the current time."
    ),
    bot: bool = Field(
        default=False,
        description="Whether to use the bot token or user token"
    )
) -> dict:
    """This function can be used to get messages from a specific channel."""

    # Enable calling this step without pydantic model_validate()
    cursor = cursor.default if hasattr(cursor, "default") else cursor
    include_all_metadata = include_all_metadata.default if hasattr(include_all_metadata, "default") else include_all_metadata
    inclusive = inclusive.default if hasattr(inclusive, "default") else inclusive
    limit = limit.default if hasattr(limit, "default") else limit
    latest = latest.default if hasattr(latest, "default") else latest
    oldest = oldest.default if hasattr(oldest, "default") else oldest

    if bot:
        token = os.getenv("SLACK_BOT_TOKEN")
    else:
        token = os.getenv("SLACK_USER_TOKEN")
    if not token:
        if bot:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN not found")
        else:
            raise APIKeyNotFoundError("SLACK_USER_TOKEN not found")
        
    try:
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
    except Exception as e:
        return f"Error creating conversation: {e}"


def send_message(
    channel: str = Field(
        description="The channel id or channel name to send message to."
        "If you want to send a message to a user, you can use the user's id."
    ),
    text: str | None = Field(
        default=None,
        description="The message to be sent"
    ),
    attachments: str | None = Field(
        default=None,
        description="A JSON-based array of structured attachments, presented as a URL-encoded string."
    ),
    blocks: str | None = Field(
        default=None,
        description="A JSON-based array of structured blocks, presented as a URL-encoded string."
    ),
    bot: bool = Field(
        default=False,
        description="Whether to use the bot token or user token."
    ),
    icon_emoji: str | None = Field(
        default=None,
        description="Emoji to use as the icon for this message. Overrides icon_url."
    ),
    icon_url: str | None = Field(
        default=None,
        description="URL to an image to use as the icon for this message."
    ),
    link_names: bool = Field(
        default=False,
        description="Find and link user groups."
        "No longer supports linking individual users; use syntax shown in Mentioning Users instead."
    ),
    markdown_text: str | None = Field(
        default=None,
        description="Accepts message text formatted in markdown. "
        "This argument should not be used in conjunction with blocks or text. "
        "Limit this field to 12,000 characters."
    ),
    metadata: str | None = Field(
        default=None,
        description="JSON object with event_type and event_payload fields, presented as a URL-encoded string."
    ),
    parse: str | None = Field(
        default=None,
        description="Change how messages are treated. "
        "Options are none or full. "
    ),
    reply_broadcast: bool = Field(
        default=False,
        description="Used in conjunction with thread_ts and indicates whether reply should be made visible to everyone in the channel or conversation. "
        "Defaults to false."
    ),
    thread_ts: str | None = Field(
        default=None,
        description="Provide another message's ts value to make this message a reply. "
        "Avoid using a reply's ts value; use its parent instead."
    ),
    unfurl_links: bool = Field(
        default=False,
        description="Pass true to enable unfurling of primarily text-based content. Default is false."
    ),
    unfurl_media: bool = Field(
        default=False,
        description="Pass true to enable unfurling of media content. Default is false."
    )
) -> dict:
    """This function can be used to send a message to a specific channel."""
    
    # Enable calling this step without pydantic model_validate()
    bot = bot.default if hasattr(bot, "default") else bot
    text = text.default if hasattr(text, "default") else text
    attachments = attachments.default if hasattr(attachments, "default") else attachments
    blocks = blocks.default if hasattr(blocks, "default") else blocks
    icon_emoji = icon_emoji.default if hasattr(icon_emoji, "default") else icon_emoji
    icon_url = icon_url.default if hasattr(icon_url, "default") else icon_url
    link_names = link_names.default if hasattr(link_names, "default") else link_names
    markdown_text = markdown_text.default if hasattr(markdown_text, "default") else markdown_text
    metadata = metadata.default if hasattr(metadata, "default") else metadata
    parse = parse.default if hasattr(parse, "default") else parse
    reply_broadcast = reply_broadcast.default if hasattr(reply_broadcast, "default") else reply_broadcast
    thread_ts = thread_ts.default if hasattr(thread_ts, "default") else thread_ts
    unfurl_links = unfurl_links.default if hasattr(unfurl_links, "default") else unfurl_links
    unfurl_media = unfurl_media.default if hasattr(unfurl_media, "default") else unfurl_media

    if bot:
        token = os.getenv("SLACK_BOT_TOKEN")
    else:
        token = os.getenv("SLACK_USER_TOKEN")
    if not token:
        if bot:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN not found")
        else:
            raise APIKeyNotFoundError("SLACK_USER_TOKEN not found")
        
    if text is None and attachments is None and blocks is None:
        raise ValueError("Either text, attachments, or blocks must be provided to send a message")
    
    if markdown_text is not None and (text is not None or blocks is not None):
        raise ValueError("markdown_text should not be used in conjunction with text or blocks")

    try:
        client = WebClient(token=token)
        result = client.chat_postMessage(
            channel=channel, 
            text=text,
            attachments=attachments,
            blocks=blocks,
            icon_emoji=icon_emoji,
            icon_url=icon_url,
            link_names=link_names,
            markdown_text=markdown_text,
            metadata=metadata,
            parse=parse,
            reply_broadcast=reply_broadcast,
            thread_ts=thread_ts,
            unfurl_links=unfurl_links,
            unfurl_media=unfurl_media
        )
        return result
    except Exception as e:
        return f"Error sending message: {e}"
    

def schedule_message(
    channel: str = Field(
        description="The channel name or ID to send message to"
    ),
    text: str | None = Field(
        default=None,
        description="The message to be sent"
    ),
    attachments: str | None = Field(
        default=None,
        description="A JSON-based array of structured attachments, presented as a URL-encoded string."
    ),
    blocks: str | None = Field(
        default=None,
        description="A JSON-based array of structured blocks, presented as a URL-encoded string."
    ),
    minutes_from_now: int | None = Field(
        default=None,
        description="Number of minutes from now to schedule the message"
    ),
    timestamp: str | None = Field(
        default=None,
        description="Unix timestamp representing the future time the message should post to Slack."
        "The datetime in format: YYYY-MM-DDTHH:MM:SS±hh:mm."
    ),
    link_names: bool = Field(
        default=False,
        description="Find and link user groups."
        "No longer supports linking individual users; use syntax shown in Mentioning Users instead."
    ),
    markdown_text: str | None = Field(
        default= None,
        description="Accepts message text formatted in markdown. "
        "This argument should not be used in conjunction with blocks or text. "
        "Limit this field to 12,000 characters."
    ),
    metadata: str | None = Field(
        default=None,
        description="JSON object with event_type and event_payload fields, presented as a URL-encoded string."
    ),
    parse: enumerate | None = Field(
        default=None,
        description="Change how messages are treated. "
        "Options are none or full. "
    ),
    reply_broadcast: bool = Field(
        default=False,
        description="Used in conjunction with thread_ts and indicates whether reply should be made visible to everyone in the channel or conversation. "
        "Defaults to false."
    ),
    thread_ts: str | None = Field(
        default=None,
        description="Provide another message's ts value to make this message a reply. "
        "Avoid using a reply's ts value; use its parent instead."
    ),
    unfurl_links: bool = Field(
        default=False,
        description="Pass true to enable unfurling of primarily text-based content. Default is false."
    ),
    unfurl_media: bool = Field(
        default=False,
        description="Pass true to enable unfurling of media content. Default is false."
    )
) -> dict:
    """This function can be used to schedule a message to be sent in a specific time or in a certain number of minutes from now
    It is only supported by the bot token."""
    
    # Enable calling this step without pydantic model_validate()
    minutes_from_now = minutes_from_now.default if hasattr(minutes_from_now, "default") else minutes_from_now
    timestamp = timestamp.default if hasattr(timestamp, "default") else timestamp
    text = text.default if hasattr(text, "default") else text
    attachments = attachments.default if hasattr(attachments, "default") else attachments
    blocks = blocks.default if hasattr(blocks, "default") else blocks
    link_names = link_names.default if hasattr(link_names, "default") else link_names
    markdown_text = markdown_text.default if hasattr(markdown_text, "default") else markdown_text
    metadata = metadata.default if hasattr(metadata, "default") else metadata
    parse = parse.default if hasattr(parse, "default") else parse
    reply_broadcast = reply_broadcast.default if hasattr(reply_broadcast, "default") else reply_broadcast
    thread_ts = thread_ts.default if hasattr(thread_ts, "default") else thread_ts
    unfurl_links = unfurl_links.default if hasattr(unfurl_links, "default") else unfurl_links
    unfurl_media = unfurl_media.default if hasattr(unfurl_media, "default") else unfurl_media

    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise APIKeyNotFoundError("SLACK_BOT_TOKEN not found")
    
    try:
        client = WebClient(token=token)
        
        if minutes_from_now is not None:
            scheduled_time = datetime.now() + timedelta(minutes=minutes_from_now)
            unix_timestamp = scheduled_time.timestamp()
        elif timestamp is not None:
            unix_timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z").timestamp()
        else:
            raise ValueError("Either minutes_from_now or timestamp must be provided to schedule a message")
        
        if text is None and attachments is None and blocks is None:
            raise ValueError("Either text, attachments, or blocks must be provided to schedule a message")
        
        if markdown_text is not None and (text is not None or blocks is not None):
            raise ValueError("markdown_text should not be used in conjunction with text or blocks")
        
        result = client.chat_scheduleMessage(
            channel=channel, 
            text=text, 
            attachments=attachments,
            blocks=blocks,
            post_at=int(unix_timestamp),
            link_names=link_names,
            markdown_text=markdown_text,
            metadata=metadata,
            parse=parse,
            reply_broadcast=reply_broadcast,
            thread_ts=thread_ts,
            unfurl_links=unfurl_links,
            unfurl_media=unfurl_media
        )
        
        return result
        
    except Exception as e:
        return f"Error scheduling message: {e}"