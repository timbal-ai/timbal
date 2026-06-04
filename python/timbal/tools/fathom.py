import os
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_FATHOM_BASE = "https://api.fathom.ai/external/v1"


async def _resolve_api_key(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        if "api_key" in credentials and credentials["api_key"]:
            return str(credentials["api_key"])
        if "token" in credentials and credentials["token"]:
            return str(credentials["token"])
        raise ValueError("Fathom integration credentials must include api_key or token.")
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("FATHOM_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Fathom API key not found. Set FATHOM_API_KEY, pass api_key in config, or configure an integration."
    )


def _fathom_headers(api_key: str) -> dict[str, str]:
    return {"X-Api-Key": api_key}


class FathomListMeetings(Tool):
    name: str = "fathom_list_meetings"
    description: str | None = (
        "List Fathom meetings with optional filters (date range, recorder emails, team names, "
        "invitee company domains) and cursor-based pagination."
    )
    integration: Annotated[str, Integration("fathom")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_meetings(
            cursor: str | None = Field(None, description="Pagination cursor from a previous response's next_cursor."),
            created_after: str | None = Field(
                None,
                description="ISO 8601 datetime; return meetings with created_at after this (e.g. 2025-01-01T00:00:00Z).",
            ),
            created_before: str | None = Field(
                None,
                description="ISO 8601 datetime; return meetings with created_at before this.",
            ),
            recorded_by: list[str] | None = Field(
                None,
                description="Email addresses of users who recorded meetings; matches any of the listed recorders.",
            ),
            teams: list[str] | None = Field(
                None, description="Team names; returns meetings belonging to any listed team."
            ),
            calendar_invitees_domains: list[str] | None = Field(
                None,
                description="Company email domains of calendar invitees to match (exact domain match).",
            ),
            calendar_invitees_domains_type: Literal["all", "only_internal", "one_or_more_external"] | None = Field(
                None,
                description="Filter by whether invitees include only internal domains or one or more external. Omit for API default (all).",
            ),
            include_summary: bool = Field(False, description="Include summary inline per meeting (API key auth only)."),
            include_transcript: bool = Field(
                False,
                description="Include transcript inline per meeting (API key auth only).",
            ),
            include_action_items: bool = Field(False, description="Include action items per meeting."),
            include_crm_matches: bool = Field(False, description="Include CRM matches per meeting."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: list[tuple[str, str]] = []
            if cursor is not None:
                params.append(("cursor", cursor))
            if created_after is not None:
                params.append(("created_after", created_after))
            if created_before is not None:
                params.append(("created_before", created_before))
            if calendar_invitees_domains_type is not None:
                params.append(("calendar_invitees_domains_type", calendar_invitees_domains_type))
            if recorded_by:
                for email in recorded_by:
                    params.append(("recorded_by[]", email))
            if teams:
                for t in teams:
                    params.append(("teams[]", t))
            if calendar_invitees_domains:
                for d in calendar_invitees_domains:
                    params.append(("calendar_invitees_domains[]", d))
            if include_summary:
                params.append(("include_summary", "true"))
            if include_transcript:
                params.append(("include_transcript", "true"))
            if include_action_items:
                params.append(("include_action_items", "true"))
            if include_crm_matches:
                params.append(("include_crm_matches", "true"))

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_FATHOM_BASE}/meetings",
                    headers=_fathom_headers(api_key),
                    params=params,
                    timeout=httpx.Timeout(30.0),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_meetings, **kwargs)


class FathomGetRecordingSummary(Tool):
    name: str = "fathom_get_recording_summary"
    description: str | None = "Get the AI-generated markdown summary for a Fathom meeting recording."
    integration: Annotated[str, Integration("fathom")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_summary(
            recording_id: int = Field(..., description="Fathom recording ID from list meetings or meeting metadata."),
            destination_url: str | None = Field(
                None,
                description="If set, Fathom POSTs the summary asynchronously to this URL instead of returning it.",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, str] = {}
            if destination_url is not None:
                params["destination_url"] = destination_url

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_FATHOM_BASE}/recordings/{recording_id}/summary",
                    headers=_fathom_headers(api_key),
                    params=params or None,
                    timeout=httpx.Timeout(60.0),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_summary, **kwargs)


class FathomGetRecordingTranscript(Tool):
    name: str = "fathom_get_recording_transcript"
    description: str | None = (
        "Get the full transcript for a Fathom recording with speaker attribution and HH:MM:SS timestamps."
    )
    integration: Annotated[str, Integration("fathom")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_transcript(
            recording_id: int = Field(..., description="Fathom recording ID from list meetings or meeting metadata."),
            destination_url: str | None = Field(
                None,
                description="If set, Fathom POSTs the transcript asynchronously to this URL instead of returning it.",
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, str] = {}
            if destination_url is not None:
                params["destination_url"] = destination_url

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_FATHOM_BASE}/recordings/{recording_id}/transcript",
                    headers=_fathom_headers(api_key),
                    params=params or None,
                    timeout=httpx.Timeout(120.0),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_transcript, **kwargs)


class FathomListTeams(Tool):
    name: str = "fathom_list_teams"
    description: str | None = "List all teams in the Fathom organization with cursor-based pagination."
    integration: Annotated[str, Integration("fathom")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_teams(
            cursor: str | None = Field(None, description="Pagination cursor from a previous response's next_cursor."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, str] = {}
            if cursor is not None:
                params["cursor"] = cursor

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_FATHOM_BASE}/teams",
                    headers=_fathom_headers(api_key),
                    params=params or None,
                    timeout=httpx.Timeout(30.0),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_teams, **kwargs)


class FathomListTeamMembers(Tool):
    name: str = "fathom_list_team_members"
    description: str | None = "List Fathom team members with optional filter by team name and cursor-based pagination."
    integration: Annotated[str, Integration("fathom")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_team_members(
            cursor: str | None = Field(None, description="Pagination cursor from a previous response's next_cursor."),
            team: str | None = Field(None, description="Filter to members of this team name."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, str] = {}
            if cursor is not None:
                params["cursor"] = cursor
            if team is not None:
                params["team"] = team

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_FATHOM_BASE}/team_members",
                    headers=_fathom_headers(api_key),
                    params=params or None,
                    timeout=httpx.Timeout(30.0),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_team_members, **kwargs)
