import asyncio
import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_SLACK_API_BASE = "https://slack.com/api"


async def _resolve_token(tool: Any) -> str:
    """Resolve Slack API token from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_token"]
    if tool.api_token is not None:
        return tool.api_token.get_secret_value()
    env_key = os.getenv("SLACK_API_TOKEN")
    if env_key:
        return env_key
    raise ValueError(
        "Slack API token not found. Set SLACK_API_TOKEN environment variable, "
        "pass api_token in config, or configure an integration."
    )


async def _resolve_token_and_channel(tool: Any, channel: str | None = None) -> tuple[str, str]:
    """Resolve Slack token and a channel ID, using parameter or integration default."""
    if channel is not None:
        token = await _resolve_token(tool)
        return token, channel

    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        token = credentials["api_token"]
        channel_id = credentials.get("channel_id")
        if not channel_id:
            raise ValueError(
                "Slack integration is missing 'channel_id'. Provide channel explicitly or configure the integration."
            )
        return token, channel_id

    raise ValueError(
        "Channel ID not provided and Slack integration has no default 'channel_id'. "
        "Provide channel explicitly or configure the integration."
    )


class ReadMessages(Tool):
    name: str = "slack_read_messages"
    description: str | None = "Read messages from a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _read_messages(
            channel: str | None = Field(None, description="Channel ID to read messages from (e.g. 'C1234567890'). If not provided, uses channel_id from integration"),
            limit: int = Field(20, description="Maximum number of messages to return"),
            cursor: str | None = Field(None, description="Pagination cursor for next page"),
            oldest: str | None = Field(None, description="Timestamp of oldest message to include"),
            latest: str | None = Field(None, description="Timestamp of latest message to include"),
            inclusive: bool = Field(False, description="Include messages with timestamps exactly matching oldest/latest"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            params: dict[str, Any] = {"channel": target_channel, "limit": limit, "inclusive": inclusive}
            if cursor:
                params["cursor"] = cursor
            if oldest:
                params["oldest"] = oldest
            if latest:
                params["latest"] = latest

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/conversations.history",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ReadMessages"

        super().__init__(handler=_read_messages, metadata=metadata, **kwargs)


class SendMessage(Tool):
    name: str = "slack_send_message"
    description: str | None = (
        "Send a message to a Slack channel."
        "Supports plain text and Block Kit UI elements. Either 'text' or 'blocks' must be provided."
    )
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_message(
            channel: str | None = Field(None, description="Slack channel ID or name (e.g., C1234567890 or general). If not provided, uses channel_id from integration"),
            text: str | None = Field(None, description="Plain text message (required if blocks not provided)"),
            blocks: list[dict[str, Any]] | None = Field(None, description="Slack Block Kit blocks for rich formatting"),
            thread_ts: str | None = Field(None, description="Thread timestamp to reply to a specific message"),
            reply_broadcast: bool = Field(False, description="Whether to broadcast reply to channel"),
            unfurl_links: bool = Field(True, description="Whether to unfurl links in the message"),
            unfurl_media: bool = Field(True, description="Whether to unfurl media in the message"),
            username: str | None = Field(None, description="Custom username for the bot"),
            icon_emoji: str | None = Field(None, description="Custom emoji icon for the bot (e.g., ':robot_face:')"),
            icon_url: str | None = Field(None, description="Custom icon URL for the bot"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            body: dict[str, Any] = {
                "channel": target_channel,
                "unfurl_links": unfurl_links,
                "unfurl_media": unfurl_media,
                "reply_broadcast": reply_broadcast,
            }
            if text:
                body["text"] = text
            if blocks:
                body["blocks"] = blocks
            if thread_ts:
                body["thread_ts"] = thread_ts
            if username:
                body["username"] = username
            if icon_emoji:
                body["icon_emoji"] = icon_emoji
            if icon_url:
                body["icon_url"] = icon_url

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/chat.postMessage",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/SendMessage"

        super().__init__(handler=_send_message, metadata=metadata, **kwargs)


class SendEphemeralMessage(Tool):
    name: str = "slack_send_ephemeral_message"
    description: str | None = "Send an ephemeral message to a user in a channel (only visible to that user)."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_ephemeral_message(
            user: str = Field(..., description="Slack user ID who will see the ephemeral message"),
            channel: str | None = Field(None, description="Slack channel ID. If not provided, uses channel_id from integration"),
            text: str | None = Field(None, description="Plain text message (required if blocks not provided)"),
            blocks: list[dict[str, Any]] | None = Field(None, description="Slack Block Kit blocks for rich formatting"),
            thread_ts: str | None = Field(None, description="Thread timestamp to reply to a specific message"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            body: dict[str, Any] = {"channel": target_channel, "user": user}
            if text:
                body["text"] = text
            if blocks:
                body["blocks"] = blocks
            if thread_ts:
                body["thread_ts"] = thread_ts

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/chat.postEphemeral",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/SendEphemeralMessage"

        super().__init__(handler=_send_ephemeral_message, metadata=metadata, **kwargs)


class CreateCanvas(Tool):
    name: str = "slack_create_canvas"
    description: str | None = "Create a Slack canvas with rich content."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_canvas(
            title: str = Field(..., description="Title of the canvas"),
            document_content: dict[str, Any] | None = Field(None, description="Canvas document content, e.g. {'type': 'markdown', 'markdown': '## Hello\nThis is a canvas.'}"),
            channel_id: str | None = Field(None, description="Channel ID to associate the canvas with. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel_id = await _resolve_token_and_channel(self, channel_id)

            body: dict[str, Any] = {"title": title}
            if document_content:
                body["document_content"] = document_content
            if target_channel_id:
                body["channel_id"] = target_channel_id

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/canvases.create",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/CreateCanvas"

        super().__init__(handler=_create_canvas, metadata=metadata, **kwargs)


class DeleteMessage(Tool):
    name: str = "slack_delete_message"
    description: str | None = "Delete a Slack message."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_message(
            ts: str = Field(..., description="Timestamp of the message to delete (the message's unique ID)"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/chat.delete",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "ts": ts},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/DeleteMessage"

        super().__init__(handler=_delete_message, metadata=metadata, **kwargs)


class GetMessageThread(Tool):
    name: str = "slack_get_message_thread"
    description: str | None = "Retrieve a message and all its replies in a thread."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_message_thread(
            ts: str = Field(..., description="Timestamp of the parent message (thread root)"),
            limit: int = Field(200, description="Maximum number of thread messages to return"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
            cursor: str | None = Field(None, description="Pagination cursor for next page"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            params: dict[str, Any] = {"channel": target_channel, "ts": ts, "limit": limit}
            if cursor:
                params["cursor"] = cursor

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/conversations.replies",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/GetMessageThread"

        super().__init__(handler=_get_message_thread, metadata=metadata, **kwargs)


class PinMessage(Tool):
    name: str = "slack_pin_message"
    description: str | None = "Pin a message in a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _pin_message(
            timestamp: str = Field(..., description="Timestamp of the message to pin"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/pins.add",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "timestamp": timestamp},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/PinMessage"

        super().__init__(handler=_pin_message, metadata=metadata, **kwargs)


class UnpinMessage(Tool):
    name: str = "slack_unpin_message"
    description: str | None = "Unpin a message from a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _unpin_message(
            timestamp: str = Field(..., description="Timestamp of the message to unpin"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/pins.remove",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "timestamp": timestamp},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UnpinMessage"

        super().__init__(handler=_unpin_message, metadata=metadata, **kwargs)


class ListPinnedItems(Tool):
    name: str = "slack_list_pinned_items"
    description: str | None = "List all pinned items in a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pinned_items(channel: str | None = Field(None, description="Channel ID to list pinned items from. If not provided, uses channel_id from integration")) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/pins.list",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"channel": target_channel},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ListPinnedItems"

        super().__init__(handler=_list_pinned_items, metadata=metadata, **kwargs)


class GetUserPresence(Tool):
    name: str = "slack_get_user_presence"
    description: str | None = "Check a Slack user's online presence status."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_user_presence(user: str = Field(..., description="Slack user ID to get presence for (e.g. 'U1234567890')")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/users.getPresence",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"user": user},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/GetUserPresence"

        super().__init__(handler=_get_user_presence, metadata=metadata, **kwargs)


class SearchUsers(Tool):
    name: str = "slack_search_users"
    description: str | None = (
        "Search for users in the Slack workspace by email, user ID, or name."
    )
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_users(
            email: str | None = Field(None, description="Slack User Email. Exact match."),
            user_id: str | None = Field(None, description="Slack User ID. Exact Lookup."),
            name: str | None = Field(None, description="Slack User Name. Fuzzy List Search."),
            limit: int = Field(20, description="Maximum number of users to return"),
            cursor: str | None = Field(None, description="Pagination cursor for next page"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                if email:
                    response = await client.get(
                        f"{_SLACK_API_BASE}/users.lookupByEmail",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"email": email},
                    )
                elif user_id:
                    response = await client.get(
                        f"{_SLACK_API_BASE}/users.info",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"user": user_id},
                    )
                else:
                    params: dict[str, Any] = {"limit": limit}
                    if name:
                        params["name"] = name
                    if cursor:
                        params["cursor"] = cursor
                    response = await client.get(
                        f"{_SLACK_API_BASE}/users.list",
                        headers={"Authorization": f"Bearer {token}"},
                        params=params,
                    )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/SearchUsers"

        super().__init__(handler=_search_users, metadata=metadata, **kwargs)


class AddUserToChannel(Tool):
    name: str = "slack_add_user_to_channel"
    description: str | None = "Add one or more users to a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_user_to_channel(
            users: list[str] = Field(..., description="List of user IDs to invite to channel (up to 1000)."),
            channel: str | None = Field(None, description="Channel ID to add users to. If no channel is specified, uses the integration's default channel.")
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.invite",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "users": ",".join(users)},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/AddUserToChannel"

        super().__init__(handler=_add_user_to_channel, metadata=metadata, **kwargs)


class RemoveFromChannel(Tool):
    name: str = "slack_remove_from_channel"
    description: str | None = "Remove a user from a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_from_channel(
            user: str = Field(..., description="User ID to remove from channel"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.kick",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "user": user},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/RemoveFromChannel"

        super().__init__(handler=_remove_from_channel, metadata=metadata, **kwargs)


class ListUsersInChannel(Tool):
    name: str = "slack_list_users_in_channel"
    description: str | None = "List all users (member IDs) in a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_users_in_channel(
            limit: int = Field(200, description="Maximum number of users to return"),
            channel: str | None = Field(None, description="Channel ID to list users from. If not provided, uses channel_id from integration"),
            cursor: str | None = Field(None, description="Pagination cursor for next page"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            params: dict[str, Any] = {"channel": target_channel, "limit": limit}
            if cursor:
                params["cursor"] = cursor

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/conversations.members",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ListUsersInChannel"

        super().__init__(handler=_list_users_in_channel, metadata=metadata, **kwargs)


class CreateChannel(Tool):
    name: str = "slack_create_channel"
    description: str | None = "Create a new public or private Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_channel(
            name: str = Field(..., description="Channel name (lowercase, no spaces - hyphens and underscores allowed)"),
            is_private: bool = Field(False, description="If True, creates a private channel"),
            team_id: str | None = Field(None, description="Required for Enterprise Grid to specify the workspace"),
        ) -> Any:
            token = await _resolve_token(self)

            body: dict[str, Any] = {"name": name, "is_private": is_private}
            if team_id:
                body["team_id"] = team_id

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.create",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/CreateChannel"

        super().__init__(handler=_create_channel, metadata=metadata, **kwargs)


class UpdateChannelTopic(Tool):
    name: str = "slack_update_channel_topic"
    description: str | None = "Set the topic for a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_channel_topic(
            topic: str = Field(..., description="New channel topic"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.setTopic",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "topic": topic},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UpdateChannelTopic"

        super().__init__(handler=_update_channel_topic, metadata=metadata, **kwargs)


class UpdateChannelPurpose(Tool):
    name: str = "slack_update_channel_purpose"
    description: str | None = "Set the purpose for a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_channel_purpose(
            purpose: str = Field(..., description="New channel purpose"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.setPurpose",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "purpose": purpose},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UpdateChannelPurpose"

        super().__init__(handler=_update_channel_purpose, metadata=metadata, **kwargs)


class ArchiveChannel(Tool):
    name: str = "slack_archive_channel"
    description: str | None = "Archive a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _archive_channel(channel: str | None = Field(None, description="Channel ID to archive. If not provided, uses channel_id from integration")) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.archive",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ArchiveChannel"

        super().__init__(handler=_archive_channel, metadata=metadata, **kwargs)


class UnarchiveChannel(Tool):
    name: str = "slack_unarchive_channel"
    description: str | None = "Unarchive a Slack channel."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _unarchive_channel(channel: str | None = Field(None, description="Channel ID to unarchive. If not provided, uses channel_id from integration")) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.unarchive",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UnarchiveChannel"

        super().__init__(handler=_unarchive_channel, metadata=metadata, **kwargs)


class GetConversationInfo(Tool):
    name: str = "slack_get_conversation_info"
    description: str | None = "Retrieve detailed information about a Slack conversation (channel, DM, or group DM)."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_conversation_info(
            include_locale: bool = Field(False, description="Whether to include locale information"),
            include_num_members: bool = Field(True, description="Whether to include member count"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/conversations.info",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "channel": target_channel,
                        "include_locale": include_locale,
                        "include_num_members": include_num_members,
                    },
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/GetConversationInfo"

        super().__init__(handler=_get_conversation_info, metadata=metadata, **kwargs)


class ListChannels(Tool):
    name: str = "slack_list_channels"
    description: str | None = "List channels in the Slack workspace."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_channels(
            types: str = Field("public_channel", description="Comma-separated list of channel types to include. Options: 'public_channel', 'private_channel', 'mpim', 'im'"),
            limit: int = Field(100, description="Maximum number of channels to return"),
            cursor: str | None = Field(None, description="Pagination cursor for next page"),
            exclude_archived: bool = Field(True, description="Whether to exclude archived channels"),
        ) -> Any:
            token = await _resolve_token(self)

            params: dict[str, Any] = {
                "types": types,
                "limit": limit,
                "exclude_archived": exclude_archived,
            }
            if cursor:
                params["cursor"] = cursor

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/conversations.list",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ListChannels"

        super().__init__(handler=_list_channels, metadata=metadata, **kwargs)


class Search(Tool):
    name: str = "slack_search"
    description: str | None = (
        "Find messages, files, channels, and people across the Slack workspace. "
        "Supports natural language queries and keyword filters."
    )
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search(
            query: str = Field(..., description="Search query. Supports modifiers: 'in:#channel' (filter by channel), 'from:@user' (filter by sender), 'before:YYYY-MM-DD'/'after:YYYY-MM-DD' (filter by date), 'has:star'/'has:reaction' (filter by reaction/star), 'is:dm'/'is:thread' (filter by conversation type)"),
            count: int = Field(20, description="Number of results to return per page"),
            page: int = Field(1, description="Page number to retrieve"),
            sort: str = Field("score", description="Sort order: 'score' (relevance), 'timestamp'"),
            sort_dir: str = Field("desc", description="Sort direction: 'desc' or 'asc'"),
            highlight: bool = Field(True, description="Whether to highlight matching terms"),
        ) -> Any:
            token = await _resolve_token(self)

            params: dict[str, Any] = {
                "query": query,
                "count": count,
                "page": page,
                "sort": sort,
                "sort_dir": sort_dir,
                "highlight": highlight,
            }

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/search.all",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/Search"

        super().__init__(handler=_search, metadata=metadata, **kwargs)


class SearchMessages(Tool):
    name: str = "slack_search_messages"
    description: str | None = (
        "Search for messages across the Slack workspace. "
        "Supports channel, user, and date filters."
    )
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_messages(
            query: str = Field(..., description="Search query. Supports modifiers: 'in:#channel' (filter by channel), 'from:@user' (filter by sender), 'before:YYYY-MM-DD'/'after:YYYY-MM-DD' (date range), 'is:thread' (threaded messages only)"),
            count: int = Field(20, description="Number of results to return per page"),
            page: int = Field(1, description="Page number to retrieve"),
            sort: str = Field("score", description="Sort order: 'score' (relevance) or 'timestamp'"),
            sort_dir: str = Field("desc", description="Sort direction: 'desc' or 'asc'"),
            highlight: bool = Field(True, description="Whether to highlight matching terms"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/search.messages",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "query": query,
                        "count": count,
                        "page": page,
                        "sort": sort,
                        "sort_dir": sort_dir,
                        "highlight": highlight,
                    },
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/SearchMessages"

        super().__init__(handler=_search_messages, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------


class AddReaction(Tool):
    name: str = "slack_add_reaction"
    description: str | None = "Add an emoji reaction to a Slack message."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_reaction(
            timestamp: str = Field(..., description="Timestamp of the message to react to"),
            name: str = Field(..., description="Emoji name without colons, e.g. 'thumbsup', 'white_check_mark'"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)
                

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/reactions.add",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "timestamp": timestamp, "name": name},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/AddReaction"

        super().__init__(handler=_add_reaction, metadata=metadata, **kwargs)


class RemoveReaction(Tool):
    name: str = "slack_remove_reaction"
    description: str | None = "Remove an emoji reaction from a Slack message."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_reaction(
            timestamp: str = Field(..., description="Timestamp of the message to remove reaction from"),
            name: str = Field(..., description="Emoji name without colons, e.g. 'thumbsup', 'white_check_mark'"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/reactions.remove",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": target_channel, "timestamp": timestamp, "name": name},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/RemoveReaction"

        super().__init__(handler=_remove_reaction, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class ListFiles(Tool):
    name: str = "slack_list_files"
    description: str | None = "List files shared in the Slack workspace with optional filtering."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_files(
            channel: str | None = Field(None, description="Filter files shared in a specific channel. If not provided, uses channel_id from integration"),
            user: str | None = Field(None, description="Filter files uploaded by a specific user"),
            types: str | None = Field(None, description="Comma-separated file types to filter by, e.g. 'images,pdfs', 'snippets', 'zips'"),
            count: int = Field(20, description="Number of files to return per page"),
            page: int = Field(1, description="Page number to retrieve"),
            ts_from: str | None = Field(None, description="Unix timestamp to filter files uploaded after this time"),
            ts_to: str | None = Field(None, description="Unix timestamp to filter files uploaded before this time"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel) if channel is not None else (
                await _resolve_token(self),
                None,
            )

            params: dict[str, Any] = {"count": count, "page": page}
            if target_channel:
                params["channel"] = target_channel
            if user:
                params["user"] = user
            if types:
                params["types"] = types
            if ts_from:
                params["ts_from"] = ts_from
            if ts_to:
                params["ts_to"] = ts_to

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/files.list",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ListFiles"

        super().__init__(handler=_list_files, metadata=metadata, **kwargs)


class GetFileInfo(Tool):
    name: str = "slack_get_file_info"
    description: str | None = "Get metadata and details for a specific Slack file."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file_info(file: str = Field(..., description="File ID to get info for (e.g. 'F1234567890')")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/files.info",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"file": file},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/GetFileInfo"

        super().__init__(handler=_get_file_info, metadata=metadata, **kwargs)


class DownloadFile(Tool):
    name: str = "slack_download_file"
    description: str | None = "Download the raw content of a Slack file and return it as base64-encoded bytes."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_file(url_private: str = Field(..., description="Private URL of file to download. (file.url_private or file.url_private_download). Returns the file content as a base64-encoded string alongside the content-type header so callers can reconstruct the file.")) -> Any:
            token = await _resolve_token(self)

            import base64
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url_private,
                    headers={"Authorization": f"Bearer {token}"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                return {
                    "content_type": response.headers.get("content-type"),
                    "content_length": response.headers.get("content-length"),
                    "data": base64.b64encode(response.content).decode(),
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/DownloadFile"

        super().__init__(handler=_download_file, metadata=metadata, **kwargs)


class UploadFile(Tool):
    name: str = "slack_upload_file"
    description: str | None = "Upload a file to Slack and optionally share it to one or more channels."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            filename: str = Field(..., description="Name to give the file, e.g. 'report.csv'"),
            content: str = Field(..., description="Text content of the file (for text-based files). For binary files, pass base64-encoded content and set filetype accordingly"),
            channels: list[str] | None = Field(None, description="List of channel IDs to share the file to after uploading. If not provided, uses channel_id from integration"),
            title: str | None = Field(None, description="File title (optional)"),
            initial_comment: str | None = Field(None, description="Initial comment to add to the file"),
            thread_ts: str | None = Field(None, description="Thread timestamp to upload file as a reply"),
            filetype: str | None = Field(None, description="Slack file type identifier, e.g. 'text', 'python', 'csv', 'json'. Omit to let Slack auto-detect from filename."),
        ) -> Any:
            token = await _resolve_token(self)

            if channels:
                target_channels = channels
            else:
                if isinstance(self.integration, Integration):
                    credentials = await self.integration.resolve()
                    channel_id = credentials.get("channel_id")
                    if not channel_id:
                        raise ValueError(
                            "Slack integration is missing 'channel_id'. Provide channels explicitly or configure the integration."
                        )
                    target_channels = [channel_id]
                else:
                    raise ValueError(
                        "No channels provided and Slack integration has no default 'channel_id'. "
                        "Provide channels explicitly or configure the integration."
                    )

            data: dict[str, Any] = {"filename": filename, "content": content}
            if target_channels:
                data["channels"] = ",".join(target_channels)
            if title:
                data["title"] = title
            if initial_comment:
                data["initial_comment"] = initial_comment
            if thread_ts:
                data["thread_ts"] = thread_ts
            if filetype:
                data["filetype"] = filetype

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/files.uploadV2",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UploadFile"

        super().__init__(handler=_upload_file, metadata=metadata, **kwargs)


class DeleteFile(Tool):
    name: str = "slack_delete_file"
    description: str | None = "Delete a file from Slack."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_file(file: str = Field(..., description="File ID to delete (e.g. 'F1234567890')")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/files.delete",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"file": file},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/DeleteFile"

        super().__init__(handler=_delete_file, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Human in the loop
# ---------------------------------------------------------------------------


class SendAndWaitForResponse(Tool):
    name: str = "slack_send_and_wait_for_response"
    description: str | None = "Send a message to a Slack channel or user and wait for a reply. Returns the first reply received within the timeout window. Use this for human-in-the-loop approval, confirmation, or input flows. Returns the reply message on success, or {'timed_out': True} if no reply arrived in time."
    integration: Annotated[str, Integration("slack")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_token": self.api_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_and_wait_for_response(
            text: str = Field(..., description="Message text to send"),
            timeout_seconds: int = Field(300, description="How long to wait for a reply before returning a timeout result (default 5 minutes)"),
            poll_interval_seconds: int = Field(5, description="How often to check for new replies in seconds (default 5 seconds)"),
            channel: str | None = Field(None, description="Channel ID. If not provided, uses channel_id from integration"),
            blocks: list[dict[str, Any]] | None = Field(None, description="Slack Block Kit blocks for rich formatting"),
        ) -> Any:
            token, target_channel = await _resolve_token_and_channel(self, channel)

            headers = {"Authorization": f"Bearer {token}"}

            import httpx

            async with httpx.AsyncClient() as client:
                send_body: dict[str, Any] = {"channel": target_channel, "text": text}
                if blocks:
                    send_body["blocks"] = blocks

                send_resp = await client.post(
                    f"{_SLACK_API_BASE}/chat.postMessage",
                    headers=headers,
                    json=send_body,
                )
                send_resp.raise_for_status()
                sent = send_resp.json()

                if not sent.get("ok"):
                    return sent

                thread_ts = sent["ts"]
                bot_user_id = sent.get("message", {}).get("bot_id")
                elapsed = 0

                while elapsed < timeout_seconds:
                    await asyncio.sleep(poll_interval_seconds)
                    elapsed += poll_interval_seconds

                    replies_resp = await client.get(
                        f"{_SLACK_API_BASE}/conversations.replies",
                        headers=headers,
                        params={"channel": target_channel, "ts": thread_ts, "limit": 10},
                    )
                    replies_resp.raise_for_status()
                    replies = replies_resp.json()

                    messages = replies.get("messages", [])
                    for msg in messages[1:]:
                        if msg.get("bot_id") != bot_user_id:
                            return {"reply": msg, "thread_ts": thread_ts, "timed_out": False}

                return {"timed_out": True, "thread_ts": thread_ts}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/SendAndWaitForResponse"

        super().__init__(handler=_send_and_wait_for_response, metadata=metadata, **kwargs)
