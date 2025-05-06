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

from slack_sdk import WebClient

from ...errors import APIKeyNotFoundError
from ...types.field import Field


def get_channel_info(
    channel: str | None = Field(
        default=None, 
        description="The channel id or channel name to get the information for. If None, returns all channels"
    ),
    include_num_members: bool = Field(
        default=True, 
        description="Set to true to include the member count for the specified conversation. Defaults to true"
    ),
    include_locale: bool = Field(
        default=False, 
        description="Set this to true to receive the locale for this conversation. Defaults to false"
    ),
    exclude_archived: bool = Field(
        default=False, 
        description="Set to true to exclude archived channels from the list. Defaults to false"
    ),
    limit: int = Field(
        default=100, 
        description="The maximum number of items to return. Defaults to 100"
    ),
    types: str = Field(
        default="public_channel, private_channel", 
        description="Mix and match channel types by providing a comma-separated list of any combination of public_channel, private_channel, mpim, im"
    ),
    team_id: str | None = Field(
        default=None, 
        description="Encoded team id to list channels in, required if token belongs to org-wide app"
    ),
    bot: bool = Field(
        default=False,
        description="Whether to use the bot token"
    )
) -> dict | list[dict]:
    """Retrieve information about Slack channels.
    This function can either get information about a specific channel or list all channels
    in a workspace, with various filtering and information options.
    """

    # Enable calling this step without pydantic model_validate()
    channel = channel.default if hasattr(channel, "default") else channel
    include_num_members = include_num_members.default if hasattr(include_num_members, "default") else include_num_members
    include_locale = include_locale.default if hasattr(include_locale, "default") else include_locale
    exclude_archived = exclude_archived.default if hasattr(exclude_archived, "default") else exclude_archived
    limit = limit.default if hasattr(limit, "default") else limit
    team_id = team_id.default if hasattr(team_id, "default") else team_id
    types = types.default if hasattr(types, "default") else types

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

    except Exception as e:
        return f"Error getting channel info: {e}"