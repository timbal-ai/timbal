"""
Slack Channel Retrieval Module

Setup:
1. Create app at https://api.slack.com/apps
2. Configure permissions.
    In order to retrieve channel information, you need to allow the following permissions:
    - channels:read (for public channels)
    - groups:read (for private channels)
3. Install app to workspace
4. Set environment variables:
   - SLACK_BOT_TOKEN: the message will be sent by the bot
   - SLACK_USER_TOKEN: the message will be sent by the user

Example Usage:
>>> get_channel_info(channel="general", include_num_members=True, include_locale=True)
"""
import os

from pydantic import Field
from slack_sdk import WebClient

from ...errors import APIKeyNotFoundError
from ...utils import resolve_default


def get_channel_info(
    channel: str | None = Field(
        None, 
        description="The channel id or channel name to get the information for. If None, returns all channels"
    ),
    include_num_members: bool = Field(
        True, 
        description="Set to true to include the member count for the specified conversation. Defaults to true"
    ),
    include_locale: bool = Field(
        False, 
        description="Set this to true to receive the locale for this conversation. Defaults to false"
    ),
    exclude_archived: bool = Field(
        False, 
        description="Set to true to exclude archived channels from the list. Defaults to false"
    ),
    limit: int = Field(
        100, 
        description="The maximum number of items to return. Defaults to 100"
    ),
    types: str = Field(
        "public_channel, private_channel", 
        description="Mix and match channel types by providing a comma-separated list of any combination of public_channel, private_channel, mpim, im"
    ),
    team_id: str | None = Field(
        None, 
        description="Encoded team id to list channels in, required if token belongs to org-wide app"
    ),
) -> dict | list[dict]:
    """Retrieve information about Slack channels.
    This function can either get information about a specific channel or list all channels
    in a workspace, with various filtering and information options.
    """
    channel = resolve_default("channel", channel)
    include_num_members = resolve_default("include_num_members", include_num_members)
    include_locale = resolve_default("include_locale", include_locale)
    exclude_archived = resolve_default("exclude_archived", exclude_archived)
    limit = resolve_default("limit", limit)
    types = resolve_default("types", types)
    team_id = resolve_default("team_id", team_id)
    
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise APIKeyNotFoundError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found")
        
    client = WebClient(token=token)
    
    if channel:
        result = client.conversations_info(
            channel=channel,
            include_num_members=include_num_members,
            include_locale=include_locale
        )
        channel = result["channel"]

        return channel
    else:
        channels = []
        cursor = None
        while True:
            result = client.conversations_list(
                exclude_archived=exclude_archived,
                types=types,
                limit=limit,
                team_id=team_id,
                cursor=cursor
            )
            channels.extend(result["channels"])
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        return channels
