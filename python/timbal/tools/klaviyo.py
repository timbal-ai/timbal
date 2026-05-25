import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_BASE = "https://a.klaviyo.com/api"
_API_REVISION = "2026-04-15"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Klaviyo private API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("KLAVIYO_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Klaviyo API key not found. Set KLAVIYO_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Klaviyo-API-Key {api_key}",
        "revision": _API_REVISION,
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
    }


def _pagination_params(page_cursor: str | None) -> dict[str, str]:
    if page_cursor:
        return {"page[cursor]": page_cursor}
    return {}


def _email_subscribe_consent() -> dict[str, Any]:
    return {"email": {"marketing": {"consent": "SUBSCRIBED"}}}


def _email_unsubscribe_consent() -> dict[str, Any]:
    return {"email": {"marketing": {"consent": "UNSUBSCRIBED"}}}


def _catalog_item_id(external_id: str) -> str:
    return f"$custom:::$default:::{external_id}"


def _json_or_accepted(response: Any) -> Any:
    """Parse JSON response; bulk jobs may return 202 with an empty body."""
    if not response.content:
        return {"status": "accepted"}
    return response.json()


class KlaviyoListProfiles(Tool):
    name: str = "klaviyo_list_profiles"
    description: str | None = "List Klaviyo profiles with optional filtering, sorting, and pagination."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_profiles(
            filter: str | None = Field(
                None,
                description=(
                    "Klaviyo filter expression, e.g. equals(email,\"user@example.com\") or "
                    "greater-than(created,2024-01-01T00:00:00Z)"
                ),
            ),
            sort: str | None = Field(None, description="Sort field, prefix with '-' for descending (e.g. '-created')"),
            page_cursor: str | None = Field(None, description="Pagination cursor from a previous response links.next"),
            fields: str | None = Field(None, description="Sparse fieldset for profiles, e.g. 'email,first_name,last_name'"),
            additional_fields: str | None = Field(
                None,
                description="Extra profile fields: 'subscriptions' and/or 'predictive_analytics' (comma-separated)",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort
            if fields:
                params["fields[profile]"] = fields
            if additional_fields:
                params["additional-fields[profile]"] = additional_fields

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/profiles",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_profiles, **kwargs)


class KlaviyoGetProfile(Tool):
    name: str = "klaviyo_get_profile"
    description: str | None = "Retrieve a Klaviyo profile by ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_profile(
            profile_id: str = Field(..., description="Klaviyo profile ID"),
            fields: str | None = Field(None, description="Sparse fieldset for profile attributes (comma-separated)"),
            additional_fields: str | None = Field(
                None,
                description="Extra fields: 'subscriptions' and/or 'predictive_analytics' (comma-separated)",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, str] = {}
            if fields:
                params["fields[profile]"] = fields
            if additional_fields:
                params["additional-fields[profile]"] = additional_fields

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/profiles/{profile_id}",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_profile, **kwargs)


class KlaviyoCreateProfile(Tool):
    name: str = "klaviyo_create_profile"
    description: str | None = "Create a new Klaviyo profile."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_profile(
            email: str | None = Field(None, description="Profile email address"),
            phone_number: str | None = Field(None, description="Phone number in E.164 format, e.g. +15005550006"),
            external_id: str | None = Field(None, description="External system identifier for this profile"),
            first_name: str | None = Field(None, description="First name"),
            last_name: str | None = Field(None, description="Last name"),
            organization: str | None = Field(None, description="Company or organization name"),
            title: str | None = Field(None, description="Job title"),
            image: str | None = Field(None, description="URL to profile image"),
            location: dict[str, Any] | None = Field(None, description="Location object (address, city, country, etc.)"),
            properties: dict[str, Any] | None = Field(None, description="Custom profile properties key/value map"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            attributes: dict[str, Any] = {}
            if email:
                attributes["email"] = email
            if phone_number:
                attributes["phone_number"] = phone_number
            if external_id:
                attributes["external_id"] = external_id
            if first_name:
                attributes["first_name"] = first_name
            if last_name:
                attributes["last_name"] = last_name
            if organization:
                attributes["organization"] = organization
            if title:
                attributes["title"] = title
            if image:
                attributes["image"] = image
            if location:
                attributes["location"] = location
            if properties:
                attributes["properties"] = properties

            payload = {"data": {"type": "profile", "attributes": attributes}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/profiles",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_profile, **kwargs)


class KlaviyoUpdateProfile(Tool):
    name: str = "klaviyo_update_profile"
    description: str | None = "Update an existing Klaviyo profile by ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_profile(
            profile_id: str = Field(..., description="Klaviyo profile ID to update"),
            email: str | None = Field(None, description="Profile email address"),
            phone_number: str | None = Field(None, description="Phone number in E.164 format"),
            external_id: str | None = Field(None, description="External system identifier"),
            first_name: str | None = Field(None, description="First name"),
            last_name: str | None = Field(None, description="Last name"),
            organization: str | None = Field(None, description="Company or organization name"),
            title: str | None = Field(None, description="Job title"),
            image: str | None = Field(None, description="URL to profile image"),
            location: dict[str, Any] | None = Field(None, description="Location object"),
            properties: dict[str, Any] | None = Field(None, description="Custom profile properties to set"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            attributes: dict[str, Any] = {}
            if email is not None:
                attributes["email"] = email
            if phone_number is not None:
                attributes["phone_number"] = phone_number
            if external_id is not None:
                attributes["external_id"] = external_id
            if first_name is not None:
                attributes["first_name"] = first_name
            if last_name is not None:
                attributes["last_name"] = last_name
            if organization is not None:
                attributes["organization"] = organization
            if title is not None:
                attributes["title"] = title
            if image is not None:
                attributes["image"] = image
            if location is not None:
                attributes["location"] = location
            if properties is not None:
                attributes["properties"] = properties

            payload = {"data": {"type": "profile", "id": profile_id, "attributes": attributes}}

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/profiles/{profile_id}",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_profile, **kwargs)


class KlaviyoGetProfileLists(Tool):
    name: str = "klaviyo_get_profile_lists"
    description: str | None = "Get list memberships for a Klaviyo profile."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_profile_lists(
            profile_id: str = Field(..., description="Klaviyo profile ID"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params = _pagination_params(page_cursor) or None

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/profiles/{profile_id}/lists",
                    headers=_headers(api_key),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_profile_lists, **kwargs)


class KlaviyoListLists(Tool):
    name: str = "klaviyo_list_lists"
    description: str | None = "List Klaviyo lists with optional filtering and pagination."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_lists(
            filter: str | None = Field(None, description="Klaviyo filter expression, e.g. equals(name,\"Newsletter\")"),
            sort: str | None = Field(None, description="Sort field (prefix with '-' for descending)"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/lists",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_lists, **kwargs)


class KlaviyoGetList(Tool):
    name: str = "klaviyo_get_list"
    description: str | None = "Retrieve a Klaviyo list by ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_list(list_id: str = Field(..., description="Klaviyo list ID")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/lists/{list_id}",
                    headers=_headers(api_key),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_list, **kwargs)


class KlaviyoCreateList(Tool):
    name: str = "klaviyo_create_list"
    description: str | None = "Create a new Klaviyo list."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_list(name: str = Field(..., description="List name")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            payload = {"data": {"type": "list", "attributes": {"name": name}}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/lists",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_list, **kwargs)


class KlaviyoAddProfilesToList(Tool):
    name: str = "klaviyo_add_profiles_to_list"
    description: str | None = "Add one or more profiles to a Klaviyo list (max 1000 per call)."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_profiles_to_list(
            list_id: str = Field(..., description="Klaviyo list ID"),
            profile_ids: list[str] = Field(..., description="Profile IDs to add (max 1000)"),
        ) -> dict[str, str]:
            api_key = await _resolve_api_key(self)
            import httpx

            payload = {
                "data": [{"type": "profile", "id": profile_id} for profile_id in profile_ids],
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/lists/{list_id}/relationships/profiles",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return {"status": "success"}

        super().__init__(handler=_add_profiles_to_list, **kwargs)


class KlaviyoRemoveProfilesFromList(Tool):
    name: str = "klaviyo_remove_profiles_from_list"
    description: str | None = "Remove one or more profiles from a Klaviyo list."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_profiles_from_list(
            list_id: str = Field(..., description="Klaviyo list ID"),
            profile_ids: list[str] = Field(..., description="Profile IDs to remove"),
        ) -> dict[str, str]:
            api_key = await _resolve_api_key(self)
            import httpx

            payload = {
                "data": [{"type": "profile", "id": profile_id} for profile_id in profile_ids],
            }

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    "DELETE",
                    f"{_API_BASE}/lists/{list_id}/relationships/profiles",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return {"status": "success"}

        super().__init__(handler=_remove_profiles_from_list, **kwargs)


class KlaviyoGetListProfiles(Tool):
    name: str = "klaviyo_get_list_profiles"
    description: str | None = "Get profiles belonging to a Klaviyo list."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_list_profiles(
            list_id: str = Field(..., description="Klaviyo list ID"),
            filter: str | None = Field(
                None,
                description="Filter profiles, e.g. equals(email,\"user@example.com\")",
            ),
            sort: str | None = Field(None, description="Sort field (prefix with '-' for descending)"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/lists/{list_id}/profiles",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_list_profiles, **kwargs)


class KlaviyoListSegments(Tool):
    name: str = "klaviyo_list_segments"
    description: str | None = "List Klaviyo segments with optional filtering and pagination."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_segments(
            filter: str | None = Field(None, description="Klaviyo filter expression"),
            sort: str | None = Field(None, description="Sort field (prefix with '-' for descending)"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/segments",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_segments, **kwargs)


class KlaviyoGetSegment(Tool):
    name: str = "klaviyo_get_segment"
    description: str | None = "Retrieve a Klaviyo segment by ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_segment(segment_id: str = Field(..., description="Klaviyo segment ID")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/segments/{segment_id}",
                    headers=_headers(api_key),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_segment, **kwargs)


class KlaviyoListEvents(Tool):
    name: str = "klaviyo_list_events"
    description: str | None = "List Klaviyo events with optional filtering, sorting, and includes."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_events(
            filter: str | None = Field(
                None,
                description=(
                    "Filter expression, e.g. and(equals(profile_id,\"PROFILE_ID\"),equals(metric_id,\"METRIC_ID\"))"
                ),
            ),
            sort: str | None = Field(None, description="Sort field, e.g. 'datetime' or '-datetime'"),
            include: str | None = Field(None, description="Related resources to include, e.g. 'profile,metric'"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort
            if include:
                params["include"] = include

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/events",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_events, **kwargs)


class KlaviyoCreateEvent(Tool):
    name: str = "klaviyo_create_event"
    description: str | None = (
        "Track a profile activity event in Klaviyo. Creates or updates the profile when identifiers are provided."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_event(
            metric_name: str = Field(..., description="Event metric name, e.g. 'Viewed Product' or 'Placed Order'"),
            profile_email: str | None = Field(None, description="Profile email (identifier)"),
            profile_phone_number: str | None = Field(None, description="Profile phone in E.164 format (identifier)"),
            profile_id: str | None = Field(None, description="Existing Klaviyo profile ID (identifier)"),
            profile_external_id: str | None = Field(None, description="External profile ID (identifier)"),
            properties: dict[str, Any] = Field(
                default_factory=dict,
                description="Event properties payload (product details, order info, etc.)",
            ),
            time: str | None = Field(None, description="ISO 8601 datetime when the event occurred"),
            value: float | None = Field(None, description="Numeric/monetary value associated with the event"),
            value_currency: str | None = Field(None, description="ISO 4217 currency code for value, e.g. 'USD'"),
            unique_id: str | None = Field(None, description="Unique ID to deduplicate events"),
        ) -> dict[str, str]:
            api_key = await _resolve_api_key(self)
            import httpx

            profile_attrs: dict[str, Any] = {}
            if profile_email:
                profile_attrs["email"] = profile_email
            if profile_phone_number:
                profile_attrs["phone_number"] = profile_phone_number
            if profile_external_id:
                profile_attrs["external_id"] = profile_external_id

            profile_data: dict[str, Any] = {"type": "profile", "attributes": profile_attrs}
            if profile_id:
                profile_data["id"] = profile_id

            event_attrs: dict[str, Any] = {
                "properties": properties,
                "metric": {"data": {"type": "metric", "attributes": {"name": metric_name}}},
                "profile": {"data": profile_data},
            }
            if time:
                event_attrs["time"] = time
            if value is not None:
                event_attrs["value"] = value
            if value_currency:
                event_attrs["value_currency"] = value_currency
            if unique_id:
                event_attrs["unique_id"] = unique_id

            payload = {"data": {"type": "event", "attributes": event_attrs}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/events",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return {"status": "accepted"}

        super().__init__(handler=_create_event, **kwargs)


class KlaviyoListCampaigns(Tool):
    name: str = "klaviyo_list_campaigns"
    description: str | None = (
        "List Klaviyo campaigns. A channel filter is required by the API (defaults to email campaigns)."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_campaigns(
            channel: str = Field(
                "email",
                description="Campaign channel filter: 'email', 'sms', or 'mobile_push'",
            ),
            filter: str | None = Field(
                None,
                description="Override the channel filter with a custom Klaviyo filter expression",
            ),
            sort: str | None = Field(None, description="Sort field (prefix with '-' for descending)"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {
                "filter": filter or f"equals(messages.channel,'{channel}')",
                **_pagination_params(page_cursor),
            }
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/campaigns",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_campaigns, **kwargs)


class KlaviyoGetCampaign(Tool):
    name: str = "klaviyo_get_campaign"
    description: str | None = "Retrieve a Klaviyo campaign by ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_campaign(campaign_id: str = Field(..., description="Klaviyo campaign ID")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/campaigns/{campaign_id}",
                    headers=_headers(api_key),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_campaign, **kwargs)


class KlaviyoListMetrics(Tool):
    name: str = "klaviyo_list_metrics"
    description: str | None = "List Klaviyo metrics (event types) with optional filtering and pagination."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_metrics(
            filter: str | None = Field(None, description="Klaviyo filter expression"),
            sort: str | None = Field(None, description="Sort field (prefix with '-' for descending)"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/metrics",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_metrics, **kwargs)


class KlaviyoListFlows(Tool):
    name: str = "klaviyo_list_flows"
    description: str | None = "List Klaviyo flows (automations) with optional filtering and pagination."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_flows(
            filter: str | None = Field(None, description="Klaviyo filter expression"),
            sort: str | None = Field(None, description="Sort field (prefix with '-' for descending)"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter
            if sort:
                params["sort"] = sort

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/flows",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_flows, **kwargs)


class KlaviyoGetFlow(Tool):
    name: str = "klaviyo_get_flow"
    description: str | None = "Retrieve a Klaviyo flow by ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_flow(flow_id: str = Field(..., description="Klaviyo flow ID")) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/flows/{flow_id}",
                    headers=_headers(api_key),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_flow, **kwargs)


class KlaviyoSubscribeProfiles(Tool):
    name: str = "klaviyo_subscribe_profiles"
    description: str | None = (
        "Subscribe profiles to email marketing (bulk job). Optionally add them to a list. "
        "Use for list-triggered flows when combined with list membership."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _subscribe_profiles(
            emails: list[str] = Field(..., description="Email addresses to subscribe (max 100 per call)"),
            list_id: str | None = Field(None, description="List ID to add subscribed profiles to"),
            profile_ids: list[str] | None = Field(
                None,
                description="Optional profile IDs (used instead of email when provided, same order as emails)",
            ),
            custom_source: str | None = Field(None, description="Custom source stored on consent records"),
            historical_import: bool = Field(
                False,
                description="Historical import; requires consented_at on each profile when true",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            profile_entries: list[dict[str, Any]] = []
            for i, email in enumerate(emails):
                entry: dict[str, Any] = {
                    "type": "profile",
                    "attributes": {
                        "email": email,
                        "subscriptions": _email_subscribe_consent(),
                    },
                }
                if profile_ids and i < len(profile_ids):
                    entry["id"] = profile_ids[i]
                profile_entries.append(entry)

            job_data: dict[str, Any] = {
                "type": "profile-subscription-bulk-create-job",
                "attributes": {
                    "profiles": {"data": profile_entries},
                    "historical_import": historical_import,
                },
            }
            if custom_source:
                job_data["attributes"]["custom_source"] = custom_source
            if list_id:
                job_data["relationships"] = {
                    "list": {"data": {"type": "list", "id": list_id}},
                }

            payload = {"data": job_data}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/profile-subscription-bulk-create-jobs",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return _json_or_accepted(response)

        super().__init__(handler=_subscribe_profiles, **kwargs)


class KlaviyoUnsubscribeProfiles(Tool):
    name: str = "klaviyo_unsubscribe_profiles"
    description: str | None = (
        "Unsubscribe profiles from email marketing (bulk job). "
        "Always pass list_id when profiles may not be on that list; omitting it can globally unsubscribe."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _unsubscribe_profiles(
            emails: list[str] = Field(..., description="Email addresses to unsubscribe"),
            list_id: str | None = Field(
                None,
                description="List scope for unsubscribe (strongly recommended to avoid global unsubscribe)",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            profile_entries: list[dict[str, Any]] = [
                {
                    "type": "profile",
                    "attributes": {
                        "email": email,
                        "subscriptions": _email_unsubscribe_consent(),
                    },
                }
                for email in emails
            ]

            job_data: dict[str, Any] = {
                "type": "profile-subscription-bulk-delete-job",
                "attributes": {"profiles": {"data": profile_entries}},
            }
            if list_id:
                job_data["relationships"] = {
                    "list": {"data": {"type": "list", "id": list_id}},
                }

            payload = {"data": job_data}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/profile-subscription-bulk-delete-jobs",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return _json_or_accepted(response)

        super().__init__(handler=_unsubscribe_profiles, **kwargs)


class KlaviyoGetProfileSubscriptions(Tool):
    name: str = "klaviyo_get_profile_subscriptions"
    description: str | None = "Get a Klaviyo profile with subscription consent details."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_profile_subscriptions(
            profile_id: str = Field(..., description="Klaviyo profile ID"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/profiles/{profile_id}",
                    headers=_headers(api_key),
                    params={"additional-fields[profile]": "subscriptions"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_profile_subscriptions, **kwargs)


class KlaviyoListCatalogItems(Tool):
    name: str = "klaviyo_list_catalog_items"
    description: str | None = "List Klaviyo catalog items with optional filtering and pagination."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_catalog_items(
            filter: str | None = Field(
                None,
                description='Filter expression, e.g. equals(external_id,"SKU-1")',
            ),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = _pagination_params(page_cursor)
            if filter:
                params["filter"] = filter

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/catalog-items",
                    headers=_headers(api_key),
                    params=params or None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_catalog_items, **kwargs)


class KlaviyoCreateCatalogItem(Tool):
    name: str = "klaviyo_create_catalog_item"
    description: str | None = "Create a catalog item in the default custom catalog."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_catalog_item(
            external_id: str = Field(..., description="Unique external ID (must not contain '/')"),
            title: str = Field(..., description="Product title"),
            description: str = Field(..., description="Product description"),
            url: str = Field(..., description="Product page URL"),
            price: float | None = Field(None, description="Product price"),
            image_full_url: str | None = Field(None, description="Full-size image URL"),
            published: bool = Field(True, description="Whether the item is published"),
            custom_metadata: dict[str, Any] | None = Field(None, description="Custom metadata JSON blob"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            attributes: dict[str, Any] = {
                "external_id": external_id,
                "title": title,
                "description": description,
                "url": url,
                "integration_type": "$custom",
                "catalog_type": "$default",
                "published": published,
            }
            if price is not None:
                attributes["price"] = price
            if image_full_url:
                attributes["image_full_url"] = image_full_url
            if custom_metadata:
                attributes["custom_metadata"] = custom_metadata

            payload = {"data": {"type": "catalog-item", "attributes": attributes}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/catalog-items",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_catalog_item, **kwargs)


class KlaviyoGetCatalogItem(Tool):
    name: str = "klaviyo_get_catalog_item"
    description: str | None = "Retrieve a catalog item by external ID or full catalog item ID."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_catalog_item(
            item_id: str = Field(
                ...,
                description='Catalog item ID or external_id (external_id is prefixed as $custom:::$default:::{id})',
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            catalog_id = item_id if item_id.startswith("$custom:::") else _catalog_item_id(item_id)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/catalog-items/{catalog_id}",
                    headers=_headers(api_key),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_catalog_item, **kwargs)


class KlaviyoCreatePlacedOrderEvent(Tool):
    name: str = "klaviyo_create_placed_order_event"
    description: str | None = (
        "Track a 'Placed Order' event for a profile. Use for revenue attribution and order-based flows."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_placed_order_event(
            profile_email: str | None = Field(None, description="Profile email (identifier)"),
            profile_id: str | None = Field(None, description="Existing Klaviyo profile ID"),
            order_id: str | None = Field(None, description="Order ID stored in event properties"),
            value: float | None = Field(None, description="Order total value"),
            value_currency: str = Field("USD", description="ISO 4217 currency code"),
            properties: dict[str, Any] = Field(
                default_factory=dict,
                description="Additional order properties (line items, discounts, etc.)",
            ),
            time: str | None = Field(None, description="ISO 8601 datetime when the order was placed"),
            unique_id: str | None = Field(None, description="Unique ID to deduplicate the order event"),
        ) -> dict[str, str]:
            api_key = await _resolve_api_key(self)
            import httpx

            event_properties = dict(properties)
            if order_id:
                event_properties["OrderId"] = order_id

            profile_attrs: dict[str, Any] = {}
            if profile_email:
                profile_attrs["email"] = profile_email
            profile_data: dict[str, Any] = {"type": "profile", "attributes": profile_attrs}
            if profile_id:
                profile_data["id"] = profile_id

            event_attrs: dict[str, Any] = {
                "properties": event_properties,
                "metric": {"data": {"type": "metric", "attributes": {"name": "Placed Order"}}},
                "profile": {"data": profile_data},
            }
            if time:
                event_attrs["time"] = time
            if value is not None:
                event_attrs["value"] = value
                event_attrs["value_currency"] = value_currency
            if unique_id:
                event_attrs["unique_id"] = unique_id

            payload = {"data": {"type": "event", "attributes": event_attrs}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/events",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return {"status": "accepted"}

        super().__init__(handler=_create_placed_order_event, **kwargs)


class KlaviyoGetProfilePredictiveAnalytics(Tool):
    name: str = "klaviyo_get_profile_predictive_analytics"
    description: str | None = "Get Klaviyo predictive analytics (CLV, churn risk, etc.) for a profile."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_profile_predictive_analytics(
            profile_id: str = Field(..., description="Klaviyo profile ID"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/profiles/{profile_id}",
                    headers=_headers(api_key),
                    params={"additional-fields[profile]": "predictive_analytics"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_profile_predictive_analytics, **kwargs)


class KlaviyoCreateCampaign(Tool):
    name: str = "klaviyo_create_campaign"
    description: str | None = "Create a draft email campaign with audiences and at least one email message."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_campaign(
            name: str = Field(..., description="Campaign name"),
            audience_list_ids: list[str] = Field(..., description="List IDs to include in the campaign audience"),
            subject: str = Field(..., description="Email subject line"),
            from_email: str = Field(..., description="Sender email address (must be configured in Klaviyo)"),
            from_label: str | None = Field(None, description="Sender display name"),
            preview_text: str | None = Field(None, description="Email preview text"),
            excluded_list_ids: list[str] | None = Field(None, description="List IDs to exclude from the audience"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            audiences: dict[str, Any] = {"included": audience_list_ids}
            if excluded_list_ids:
                audiences["excluded"] = excluded_list_ids

            content: dict[str, Any] = {"subject": subject, "from_email": from_email}
            if from_label:
                content["from_label"] = from_label
            if preview_text:
                content["preview_text"] = preview_text

            payload = {
                "data": {
                    "type": "campaign",
                    "attributes": {
                        "name": name,
                        "audiences": audiences,
                        "campaign-messages": {
                            "data": [
                                {
                                    "type": "campaign-message",
                                    "attributes": {
                                        "definition": {
                                            "channel": "email",
                                            "content": content,
                                        },
                                    },
                                },
                            ],
                        },
                    },
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/campaigns",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_campaign, **kwargs)


class KlaviyoUpdateCampaign(Tool):
    name: str = "klaviyo_update_campaign"
    description: str | None = "Update a Klaviyo campaign (name, audiences, or send strategy)."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_campaign(
            campaign_id: str = Field(..., description="Klaviyo campaign ID"),
            name: str | None = Field(None, description="New campaign name"),
            audience_list_ids: list[str] | None = Field(None, description="List IDs to include in the audience"),
            excluded_list_ids: list[str] | None = Field(None, description="List IDs to exclude from the audience"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            attributes: dict[str, Any] = {}
            if name is not None:
                attributes["name"] = name
            if audience_list_ids is not None or excluded_list_ids is not None:
                audiences: dict[str, Any] = {}
                if audience_list_ids is not None:
                    audiences["included"] = audience_list_ids
                if excluded_list_ids is not None:
                    audiences["excluded"] = excluded_list_ids
                attributes["audiences"] = audiences

            payload = {
                "data": {
                    "type": "campaign",
                    "id": campaign_id,
                    "attributes": attributes,
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/campaigns/{campaign_id}",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_campaign, **kwargs)


class KlaviyoSendCampaign(Tool):
    name: str = "klaviyo_send_campaign"
    description: str | None = (
        "Trigger a campaign send job. The campaign must be in a sendable state with valid content and audiences."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_campaign(
            campaign_id: str = Field(..., description="Klaviyo campaign ID to send"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            payload = {"data": {"type": "campaign-send-job", "id": campaign_id}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/campaign-send-jobs",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return _json_or_accepted(response)

        super().__init__(handler=_send_campaign, **kwargs)


class KlaviyoQueryCampaignValues(Tool):
    name: str = "klaviyo_query_campaign_values"
    description: str | None = "Query campaign performance statistics for a timeframe and conversion metric."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query_campaign_values(
            conversion_metric_id: str = Field(..., description="Metric ID used for conversion statistics"),
            statistics: list[str] = Field(
                default=["opens", "open_rate", "clicks", "recipients"],
                description="Statistics to return (e.g. opens, click_rate, conversion_rate)",
            ),
            timeframe_key: str = Field(
                "last_7_days",
                description=(
                    "Predefined timeframe: last_7_days, last_30_days, last_90_days, today, yesterday, "
                    "this_month, last_month, etc."
                ),
            ),
            filter: str | None = Field(
                None,
                description='Optional filter, e.g. equals(campaign_id,"01GMRWDSA0ARTAKE1SFX8JGXAY")',
            ),
            group_by: list[str] | None = Field(None, description="Group-by attributes (campaign_id, send_channel, etc.)"),
            page_cursor: str | None = Field(None, description="Pagination cursor for large result sets"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            attributes: dict[str, Any] = {
                "statistics": statistics,
                "timeframe": {"key": timeframe_key},
                "conversion_metric_id": conversion_metric_id,
            }
            if filter:
                attributes["filter"] = filter
            if group_by:
                attributes["group_by"] = group_by

            params = _pagination_params(page_cursor) or None
            payload = {
                "data": {
                    "type": "campaign-values-report",
                    "attributes": attributes,
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/campaign-values-reports",
                    headers=_headers(api_key),
                    json=payload,
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_query_campaign_values, **kwargs)


class KlaviyoUpdateFlowStatus(Tool):
    name: str = "klaviyo_update_flow_status"
    description: str | None = "Update a flow's status to draft, manual, or live."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_flow_status(
            flow_id: str = Field(..., description="Klaviyo flow ID"),
            status: str = Field(
                ...,
                description="New flow status: 'draft', 'manual', or 'live'",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            payload = {
                "data": {
                    "type": "flow",
                    "id": flow_id,
                    "attributes": {"status": status},
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/flows/{flow_id}",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_flow_status, **kwargs)


class KlaviyoListFlowActions(Tool):
    name: str = "klaviyo_list_flow_actions"
    description: str | None = "List actions (steps) for a Klaviyo flow."
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_flow_actions(
            flow_id: str = Field(..., description="Klaviyo flow ID"),
            page_cursor: str | None = Field(None, description="Pagination cursor"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params = _pagination_params(page_cursor) or None

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/flows/{flow_id}/flow-actions",
                    headers=_headers(api_key),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_flow_actions, **kwargs)


class KlaviyoTriggerFlowViaEvent(Tool):
    name: str = "klaviyo_trigger_flow_via_event"
    description: str | None = (
        "Track a custom metric event to trigger metric-based flows. "
        "The metric name must match the flow trigger configuration."
    )
    integration: Annotated[str, Integration("klaviyo")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _trigger_flow_via_event(
            metric_name: str = Field(..., description="Metric name matching the flow trigger"),
            profile_email: str | None = Field(None, description="Profile email (identifier)"),
            profile_id: str | None = Field(None, description="Existing Klaviyo profile ID"),
            properties: dict[str, Any] = Field(default_factory=dict, description="Event properties"),
            unique_id: str | None = Field(None, description="Unique ID to deduplicate events"),
        ) -> dict[str, str]:
            api_key = await _resolve_api_key(self)
            import httpx

            profile_attrs: dict[str, Any] = {}
            if profile_email:
                profile_attrs["email"] = profile_email
            profile_data: dict[str, Any] = {"type": "profile", "attributes": profile_attrs}
            if profile_id:
                profile_data["id"] = profile_id

            event_attrs: dict[str, Any] = {
                "properties": properties,
                "metric": {"data": {"type": "metric", "attributes": {"name": metric_name}}},
                "profile": {"data": profile_data},
            }
            if unique_id:
                event_attrs["unique_id"] = unique_id

            payload = {"data": {"type": "event", "attributes": event_attrs}}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/events",
                    headers=_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return {"status": "accepted"}

        super().__init__(handler=_trigger_flow_via_event, **kwargs)
