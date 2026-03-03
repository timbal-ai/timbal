from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_SLACK_API_BASE = "https://slack.com/api"


class ReadMessages(Tool):
    name: str = "slack_read_messages"
    description: str | None = "Read messages from a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _read_messages(
            channel: str,
            limit: int = 20,
            cursor: str | None = None,
            oldest: str | None = None,
            latest: str | None = None,
            inclusive: bool = False,
        ) -> Any:
            """
            channel: channel ID (e.g. "C1234567890").
            cursor: pagination cursor from a previous response's response_metadata.next_cursor.
            oldest / latest: Unix timestamps to bound the time range.
            inclusive: include messages at the oldest/latest boundaries.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"channel": channel, "limit": limit, "inclusive": inclusive}
            if cursor:
                params["cursor"] = cursor
            if oldest:
                params["oldest"] = oldest
            if latest:
                params["latest"] = latest

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
        "Send a message to a Slack channel or user. "
        "Supports plain text and Block Kit UI elements. Either 'text' or 'blocks' must be provided."
    )
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_message(
            channel: str,
            text: str | None = None,
            blocks: list[dict[str, Any]] | None = None,
            thread_ts: str | None = None,
            reply_broadcast: bool = False,
            unfurl_links: bool = True,
            unfurl_media: bool = True,
            username: str | None = None,
            icon_emoji: str | None = None,
            icon_url: str | None = None,
        ) -> Any:
            """
            channel: channel ID or user ID for a DM (e.g. "C1234567890" or "U1234567890").
            text: plain text fallback (required if blocks not provided).
            blocks: Block Kit layout elements array.
            thread_ts: timestamp of parent message to reply in a thread.
            reply_broadcast: also send reply to the channel when threading.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {
                "channel": channel,
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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_ephemeral_message(
            channel: str,
            user: str,
            text: str | None = None,
            blocks: list[dict[str, Any]] | None = None,
            thread_ts: str | None = None,
        ) -> Any:
            """
            user: Slack user ID who will see the ephemeral message.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"channel": channel, "user": user}
            if text:
                body["text"] = text
            if blocks:
                body["blocks"] = blocks
            if thread_ts:
                body["thread_ts"] = thread_ts

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_canvas(
            title: str,
            document_content: dict[str, Any] | None = None,
            channel_id: str | None = None,
        ) -> Any:
            """
            title: title of the canvas.
            document_content: canvas document content, e.g.:
              {"type": "markdown", "markdown": "## Hello\\nThis is a canvas."}
            channel_id: if provided, associates the canvas with a channel.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"title": title}
            if document_content:
                body["document_content"] = document_content
            if channel_id:
                body["channel_id"] = channel_id

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_message(channel: str, ts: str) -> Any:
            """
            ts: timestamp of the message to delete (the message's unique ID).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/chat.delete",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "ts": ts},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/DeleteMessage"

        super().__init__(handler=_delete_message, metadata=metadata, **kwargs)


class GetMessageThread(Tool):
    name: str = "slack_get_message_thread"
    description: str | None = "Retrieve a message and all its replies in a thread."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_message_thread(
            channel: str,
            ts: str,
            limit: int = 200,
            cursor: str | None = None,
        ) -> Any:
            """
            ts: timestamp of the parent message (thread root).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"channel": channel, "ts": ts, "limit": limit}
            if cursor:
                params["cursor"] = cursor

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _pin_message(channel: str, timestamp: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/pins.add",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "timestamp": timestamp},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/PinMessage"

        super().__init__(handler=_pin_message, metadata=metadata, **kwargs)


class UnpinMessage(Tool):
    name: str = "slack_unpin_message"
    description: str | None = "Unpin a message from a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _unpin_message(channel: str, timestamp: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/pins.remove",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "timestamp": timestamp},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UnpinMessage"

        super().__init__(handler=_unpin_message, metadata=metadata, **kwargs)


class ListPinnedItems(Tool):
    name: str = "slack_list_pinned_items"
    description: str | None = "List all pinned items in a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pinned_items(channel: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/pins.list",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"channel": channel},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ListPinnedItems"

        super().__init__(handler=_list_pinned_items, metadata=metadata, **kwargs)


class GetUserPresence(Tool):
    name: str = "slack_get_user_presence"
    description: str | None = "Check a Slack user's online presence status."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_user_presence(user: str) -> Any:
            """
            user: Slack user ID (e.g. "U1234567890").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_users(
            email: str | None = None,
            user_id: str | None = None,
            name: str | None = None,
            limit: int = 20,
            cursor: str | None = None,
        ) -> Any:
            """
            Provide one of: email (exact match), user_id (exact lookup), or name (fuzzy list search).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_user_to_channel(channel: str, users: list[str]) -> Any:
            """
            users: list of Slack user IDs to invite (up to 1000).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.invite",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "users": ",".join(users)},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/AddUserToChannel"

        super().__init__(handler=_add_user_to_channel, metadata=metadata, **kwargs)


class RemoveFromChannel(Tool):
    name: str = "slack_remove_from_channel"
    description: str | None = "Remove a user from a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_from_channel(channel: str, user: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.kick",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "user": user},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/RemoveFromChannel"

        super().__init__(handler=_remove_from_channel, metadata=metadata, **kwargs)


class ListUsersInChannel(Tool):
    name: str = "slack_list_users_in_channel"
    description: str | None = "List all users (member IDs) in a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_users_in_channel(
            channel: str,
            limit: int = 200,
            cursor: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"channel": channel, "limit": limit}
            if cursor:
                params["cursor"] = cursor

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_channel(
            name: str,
            is_private: bool = False,
            team_id: str | None = None,
        ) -> Any:
            """
            name: channel name (lowercase, no spaces — hyphens and underscores allowed).
            is_private: if True, creates a private channel.
            team_id: required for Enterprise Grid to specify the workspace.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            body: dict[str, Any] = {"name": name, "is_private": is_private}
            if team_id:
                body["team_id"] = team_id

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
    description: str | None = "Update the topic of a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_channel_topic(channel: str, topic: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.setTopic",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "topic": topic},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UpdateChannelTopic"

        super().__init__(handler=_update_channel_topic, metadata=metadata, **kwargs)


class UpdateChannelPurpose(Tool):
    name: str = "slack_update_channel_purpose"
    description: str | None = "Update the purpose of a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_channel_purpose(channel: str, purpose: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.setPurpose",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "purpose": purpose},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UpdateChannelPurpose"

        super().__init__(handler=_update_channel_purpose, metadata=metadata, **kwargs)


class ArchiveChannel(Tool):
    name: str = "slack_archive_channel"
    description: str | None = "Archive a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _archive_channel(channel: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.archive",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/ArchiveChannel"

        super().__init__(handler=_archive_channel, metadata=metadata, **kwargs)


class UnarchiveChannel(Tool):
    name: str = "slack_unarchive_channel"
    description: str | None = "Unarchive a Slack channel."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _unarchive_channel(channel: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/conversations.unarchive",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/UnarchiveChannel"

        super().__init__(handler=_unarchive_channel, metadata=metadata, **kwargs)


class GetConversationInfo(Tool):
    name: str = "slack_get_conversation_info"
    description: str | None = "Retrieve detailed information about a Slack conversation (channel, DM, or group DM)."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_conversation_info(
            channel: str,
            include_locale: bool = False,
            include_num_members: bool = True,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_SLACK_API_BASE}/conversations.info",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "channel": channel,
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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_channels(
            types: str = "public_channel",
            limit: int = 100,
            cursor: str | None = None,
            exclude_archived: bool = True,
        ) -> Any:
            """
            types: comma-separated list of channel types to include.
                   Options: "public_channel", "private_channel", "mpim", "im".
            exclude_archived: whether to exclude archived channels.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {
                "types": types,
                "limit": limit,
                "exclude_archived": exclude_archived,
            }
            if cursor:
                params["cursor"] = cursor

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search(
            query: str,
            count: int = 20,
            page: int = 1,
            sort: str = "score",
            sort_dir: str = "desc",
            highlight: bool = True,
        ) -> Any:
            """
            query: search query. Supports modifiers:
              - "in:#channel" — filter by channel
              - "from:@user" — filter by sender
              - "before:YYYY-MM-DD" / "after:YYYY-MM-DD" — filter by date
              - "has:star", "has:reaction" — filter by reaction/star
              - "is:dm", "is:thread" — filter by conversation type
            sort: "score" (relevance) or "timestamp".
            sort_dir: "asc" or "desc".
            highlight: wrap matched terms in highlight markers in results.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {
                "query": query,
                "count": count,
                "page": page,
                "sort": sort,
                "sort_dir": sort_dir,
                "highlight": highlight,
            }

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_messages(
            query: str,
            count: int = 20,
            page: int = 1,
            sort: str = "score",
            sort_dir: str = "desc",
            highlight: bool = True,
        ) -> Any:
            """
            query: search query. Supports modifiers:
              - "in:#channel" — filter by channel
              - "from:@user" — filter by sender
              - "before:YYYY-MM-DD" / "after:YYYY-MM-DD" — date range
              - "is:thread" — threaded messages only
            sort: "score" (relevance) or "timestamp".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_reaction(channel: str, timestamp: str, name: str) -> Any:
            """
            name: emoji name without colons, e.g. "thumbsup", "white_check_mark".
            timestamp: timestamp of the message to react to.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/reactions.add",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "timestamp": timestamp, "name": name},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Slack/AddReaction"

        super().__init__(handler=_add_reaction, metadata=metadata, **kwargs)


class RemoveReaction(Tool):
    name: str = "slack_remove_reaction"
    description: str | None = "Remove an emoji reaction from a Slack message."
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_reaction(channel: str, timestamp: str, name: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/reactions.remove",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"channel": channel, "timestamp": timestamp, "name": name},
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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_files(
            channel: str | None = None,
            user: str | None = None,
            types: str | None = None,
            count: int = 20,
            page: int = 1,
            ts_from: str | None = None,
            ts_to: str | None = None,
        ) -> Any:
            """
            channel: filter files shared in a specific channel.
            user: filter files uploaded by a specific user.
            types: comma-separated file types to filter by, e.g. "images,pdfs", "snippets", "zips".
            ts_from / ts_to: Unix timestamps to bound the upload date range.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"count": count, "page": page}
            if channel:
                params["channel"] = channel
            if user:
                params["user"] = user
            if types:
                params["types"] = types
            if ts_from:
                params["ts_from"] = ts_from
            if ts_to:
                params["ts_to"] = ts_to

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file_info(file: str) -> Any:
            """
            file: Slack file ID (e.g. "F1234567890").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_file(url_private: str) -> Any:
            """
            url_private: the private download URL from a file's metadata
                         (file.url_private or file.url_private_download).
            Returns the file content as a base64-encoded string alongside
            the content-type header so callers can reconstruct the file.
            """
            import base64

            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            filename: str,
            content: str,
            channels: list[str] | None = None,
            title: str | None = None,
            initial_comment: str | None = None,
            thread_ts: str | None = None,
            filetype: str | None = None,
        ) -> Any:
            """
            filename: name to give the file, e.g. "report.csv".
            content: text content of the file (for text-based files).
                     For binary files, pass base64-encoded content and set filetype accordingly.
            channels: list of channel IDs to share the file to after uploading.
            filetype: Slack file type identifier, e.g. "text", "python", "csv", "json".
                      Omit to let Slack auto-detect from filename.
            thread_ts: post the file as a reply in this thread.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {"filename": filename, "content": content}
            if channels:
                data["channels"] = ",".join(channels)
            if title:
                data["title"] = title
            if initial_comment:
                data["initial_comment"] = initial_comment
            if thread_ts:
                data["thread_ts"] = thread_ts
            if filetype:
                data["filetype"] = filetype

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
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_file(file: str) -> Any:
            """
            file: Slack file ID (e.g. "F1234567890").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
    description: str | None = (
        "Send a message to a Slack channel or user and wait for a reply. "
        "Returns the first reply received within the timeout window. "
        "Use this for human-in-the-loop approval, confirmation, or input flows."
    )
    integration: Annotated[str, Integration("slack")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_and_wait_for_response(
            channel: str,
            text: str,
            timeout_seconds: int = 300,
            poll_interval_seconds: int = 5,
            blocks: list[dict[str, Any]] | None = None,
        ) -> Any:
            """
            Sends a message and polls the thread for a reply from any user other than the bot.
            timeout_seconds: how long to wait for a reply before returning a timeout result (default 5 min).
            poll_interval_seconds: how often to check for new replies (default 5 s).
            Returns the reply message on success, or {"timed_out": True} if no reply arrived in time.
            """
            import asyncio

            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            headers = {"Authorization": f"Bearer {token}"}

            async with httpx.AsyncClient() as client:
                send_body: dict[str, Any] = {"channel": channel, "text": text}
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
                        params={"channel": channel, "ts": thread_ts, "limit": 10},
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
