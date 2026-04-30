"""Jira Cloud REST API tools.

Auth (first match wins when deciding how to call the API):

- **Site / API token** (same style as ``tools.py``): set ``JIRA_BASE_URL`` (e.g.
  ``https://your-site.atlassian.net``), ``JIRA_API_TOKEN``, and ``JIRA_EMAIL`` or
  ``JIRA_USER_EMAIL``. Requests go to ``{JIRA_BASE_URL}/rest/api/3/...`` with Basic auth.

- **OAuth 2.0 (3LO)**: ``JIRA_ACCESS_TOKEN`` / ``ATLASSIAN_ACCESS_TOKEN`` and optional
  ``JIRA_CLOUD_ID`` / ``ATLASSIAN_CLOUD_ID``; base URL is
  ``https://api.atlassian.com/ex/jira/{cloudId}/rest/api/3/...``.

Service desk: .../rest/servicedeskapi/...
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from timbal.core import Tool



def _env_strip(key: str, default: str = "") -> str:
    raw = os.getenv(key, default) or default
    s = str(raw).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _jira_site_auth_triple() -> tuple[str, str, str] | None:
    """Return (base_url, email, api_token) when site auth env is complete."""
    base = _env_strip("JIRA_BASE_URL").rstrip("/")
    token = _env_strip("JIRA_API_TOKEN")
    email = _env_strip("JIRA_EMAIL") or _env_strip("JIRA_USER_EMAIL")
    if base and token and email:
        return (base, email, token)
    return None


def _jira_site_basic_headers(email: str, api_token: str) -> dict[str, str]:
    auth_raw = f"{email}:{api_token}"
    auth_b64 = base64.b64encode(auth_raw.encode()).decode("ascii")
    return {
        "Authorization": f"Basic {auth_b64}",
        "Accept": "application/json",
    }


async def _jira_use_site_auth(tool: Any) -> bool:
    """Use Jira site URL + email/API token when configured and no OAuth override."""
    if getattr(tool, "token", None) is not None:
        return False
    integ = getattr(tool, "integration", None)
    if integ is not None and hasattr(integ, "resolve") and callable(integ.resolve):
        try:
            creds = await integ.resolve()
        except Exception:
            creds = {}
        if isinstance(creds, dict):
            t = creds.get("token") or creds.get("access_token")
            if isinstance(t, str) and t.strip():
                return False
    return _jira_site_auth_triple() is not None


async def _jira_connection(
    tool: Any, cloud_id: str | None, site_name: str | None
) -> tuple[str, dict[str, str]]:
    """REST API base URL (no path) and headers for this request."""
    if await _jira_use_site_auth(tool):
        triple = _jira_site_auth_triple()
        assert triple is not None
        base, email, token = triple
        return base, _jira_site_basic_headers(email, token)
    cid = await _resolve_cloud_id(tool, cloud_id, site_name)
    tok = await _resolve_token(tool)
    return _jira_api_root(cid), {"Authorization": f"Bearer {tok}", "Accept": "application/json"}


class Integration:
    """Metadata stub for `Annotated[..., Integration("jira")]`; platform may supply a real object."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

_ACCESSIBLE_RESOURCES = "https://api.atlassian.com/oauth/token/accessible-resources"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        token = credentials.get("token") or credentials.get("access_token")
        if isinstance(token, str) and token.strip():
            return token.strip()
    if getattr(tool, "token", None) is not None:
        return tool.token.get_secret_value()
    for env_key in ("JIRA_ACCESS_TOKEN", "ATLASSIAN_ACCESS_TOKEN"):
        env_token = os.getenv(env_key)
        if env_token:
            return env_token
    raise ValueError(
        "Jira credentials not found. Configure a jira integration, pass token, "
        "or set JIRA_ACCESS_TOKEN / ATLASSIAN_ACCESS_TOKEN, "
        "or use site auth: JIRA_BASE_URL, JIRA_API_TOKEN, and JIRA_EMAIL (or JIRA_USER_EMAIL)."
    )


async def _fetch_accessible_resources(token: str) -> list[dict[str, Any]]:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(
            _ACCESSIBLE_RESOURCES,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        if response.status_code == 401:
            snippet = (response.text or "")[:400]
            raise ValueError(
                "Atlassian returned 401 for oauth/token/accessible-resources. "
                "These tools expect an OAuth 2.0 (3LO) access token as Bearer "
                "(JIRA_ACCESS_TOKEN / ATLASSIAN_ACCESS_TOKEN), not a Jira Cloud API token alone. "
                "Alternatively set JIRA_CLOUD_ID (or ATLASSIAN_CLOUD_ID) to skip this discovery call "
                "if your token is valid for /ex/jira/{cloudId}/rest/api only. "
                f"Response: {snippet!r}"
            )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []


async def _resolve_cloud_id(tool: Any, cloud_id: str | None, site_name: str | None) -> str:
    if cloud_id:
        return cloud_id
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        cid = credentials.get("cloud_id")
        if isinstance(cid, str) and cid.strip():
            return cid.strip()
    env_cid = os.getenv("JIRA_CLOUD_ID") or os.getenv("ATLASSIAN_CLOUD_ID")
    if isinstance(env_cid, str) and env_cid.strip():
        return env_cid.strip()
    token = await _resolve_token(tool)
    resources = await _fetch_accessible_resources(token)
    if not resources:
        raise ValueError("No accessible Jira cloud sites for this token.")
    if len(resources) == 1:
        rid = resources[0].get("id")
        if not isinstance(rid, str):
            raise ValueError("Accessible site response missing cloud id.")
        return rid
    if site_name:
        needle = site_name.strip().lower()
        for res in resources:
            name = (res.get("name") or "").lower()
            url = (res.get("url") or "").lower()
            if name == needle or needle in url or url.rstrip("/").endswith(needle):
                rid = res.get("id")
                if isinstance(rid, str):
                    return rid
        names = [(r.get("name"), r.get("url"), r.get("id")) for r in resources]
        raise ValueError(f"No site matched site_name={site_name!r}. Available: {names}")
    names = [(r.get("name"), r.get("id")) for r in resources]
    raise ValueError(
        f"Multiple Jira sites are available for this token. Pass cloud_id or site_name. Available: {names}"
    )


def _jira_api_root(cloud_id: str) -> str:
    return f"https://api.atlassian.com/ex/jira/{cloud_id}"


async def _jira_request(
    tool: Any,
    method: str,
    cloud_id: str | None,
    site_name: str | None,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    data: Any = None,
    content: bytes | str | None = None,
    headers: dict[str, str] | None = None,
    files: Any = None,
) -> Any:
    import httpx

    base, hdrs = await _jira_connection(tool, cloud_id, site_name)
    url = f"{base}{path}"
    if headers:
        hdrs = {**hdrs, **headers}
    async with httpx.AsyncClient() as client:
        kwargs: dict[str, Any] = {"method": method, "url": url, "headers": hdrs, "params": params}
        if files is not None:
            kwargs["files"] = files
            kwargs["data"] = data
        elif json_body is not None:
            kwargs["json"] = json_body
        elif content is not None:
            kwargs["content"] = content
        elif data is not None:
            kwargs["data"] = data
        response = await client.request(**kwargs)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        ctype = response.headers.get("content-type", "")
        if "application/json" in ctype:
            return response.json()
        return response.text


# Default fields when callers omit them (enhanced search defaults to ids-only without fields).
_JQL_SEARCH_DEFAULT_FIELDS = (
    "summary,status,assignee,reporter,updated,priority,issuetype,created"
)


async def _jira_search_jql_legacy_shape(
    tool: Any,
    cloud_id: str | None,
    site_name: str | None,
    *,
    jql: str,
    start_at: int = 0,
    max_results: int = 50,
    fields: str | None = None,
    expand: str | None = None,
) -> dict[str, Any]:
    """GET ``/rest/api/3/search/jql`` (replaces removed GET ``/rest/api/3/search`` — CHANGE-2046).

    Paginates with ``nextPageToken`` until enough issues exist to satisfy ``start_at`` + ``max_results``,
    then returns a dict shaped like the old search API: ``issues``, ``total`` (best-effort).
    """
    eff_fields = fields if fields is not None else _JQL_SEARCH_DEFAULT_FIELDS
    target_end = max(0, int(start_at)) + max(1, int(max_results))
    collected: list[dict[str, Any]] = []
    next_token: str | None = None
    total: int | None = None

    while len(collected) < target_end:
        need = target_end - len(collected)
        page_size = min(100, max(need, 1))
        params: dict[str, Any] = {"jql": jql, "maxResults": page_size, "fields": eff_fields}
        if expand:
            params["expand"] = expand
        if next_token:
            params["nextPageToken"] = next_token

        data = await _jira_request(
            tool,
            "GET",
            cloud_id,
            site_name,
            "/rest/api/3/search/jql",
            params=params,
        )
        if not isinstance(data, dict):
            break
        if total is None:
            raw_total = data.get("total")
            if isinstance(raw_total, int):
                total = raw_total
            else:
                for key in ("issueTotal", "totalIssues"):
                    v = data.get(key)
                    if isinstance(v, int):
                        total = v
                        break

        batch = data.get("issues") if isinstance(data.get("issues"), list) else []
        collected.extend(batch)

        if data.get("isLast"):
            break
        next_token = data.get("nextPageToken") if isinstance(data.get("nextPageToken"), str) else None
        if not next_token:
            break
        if not batch:
            break

    window = collected[max(0, int(start_at)) : target_end]
    out: dict[str, Any] = {"issues": window}
    out["total"] = total if isinstance(total, int) else len(collected)
    return out


class JiraCreateProject(Tool):
    name: str = "jira_create_project"
    description: str | None = "Create a new Jira project. With multiple Jira sites, pass cloud_id or site_name."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            key: str = Field(..., description="Project key (uppercase, e.g. ENG)."),
            name: str = Field(..., description="Display name of the project."),
            project_type_key: str = Field(
                ...,
                description="Project type: software, service_desk, or business.",
            ),
            lead_account_id: str = Field(..., description="Atlassian account id of the project lead."),
            description: str | None = Field(None, description="Optional project description."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(
                None, description="Site name or URL fragment to pick a cloud when multiple sites exist."
            ),
        ) -> Any:
            body: dict[str, Any] = {
                "key": key,
                "name": name,
                "projectTypeKey": project_type_key,
                "leadAccountId": lead_account_id,
            }
            if description is not None:
                body["description"] = description
            return await _jira_request(self, "POST", cloud_id, site_name, "/rest/api/3/project", json_body=body)

        super().__init__(handler=_handler, **kwargs)


class JiraGetProject(Tool):
    name: str = "jira_get_project"
    description: str | None = "Get metadata for a Jira project by id or key."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            project_id_or_key: str = Field(..., description="Project id or key."),
            expand: str | None = Field(None, description="Optional expand parameter per Jira API."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {}
            if expand:
                params["expand"] = expand
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/api/3/project/{project_id_or_key}",
                params=params or None,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraUpdateProject(Tool):
    name: str = "jira_update_project"
    description: str | None = "Update project fields such as name, lead, or description."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            project_id_or_key: str = Field(..., description="Project id or key."),
            fields: dict[str, Any] = Field(
                ...,
                description="Fields to update (Jira REST shape), e.g. {'name': '...', 'description': '...', 'leadAccountId': '...'}.",
            ),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(
                self,
                "PUT",
                cloud_id,
                site_name,
                f"/rest/api/3/project/{project_id_or_key}",
                json_body=fields,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraDeleteProject(Tool):
    name: str = "jira_delete_project"
    description: str | None = "Delete a project (and optionally its issues) from Jira."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            project_id_or_key: str = Field(..., description="Project id or key."),
            enable_undo: bool | None = Field(
                None,
                description="If true, Jira may retain data for undo (Cloud-specific query).",
            ),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params: dict[str, Any] = {}
            if enable_undo is not None:
                params["enableUndo"] = str(enable_undo).lower()
            return await _jira_request(
                self,
                "DELETE",
                cloud_id,
                site_name,
                f"/rest/api/3/project/{project_id_or_key}",
                params=params or None,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraListProjects(Tool):
    name: str = "jira_list_projects"
    description: str | None = "Search and list Jira projects with pagination."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            query: str | None = Field(None, description="Optional case-insensitive substring match on name or key."),
            expand: str | None = Field(None, description="Optional expand values."),
            start_at: int = Field(0, description="Pagination start index."),
            max_results: int = Field(50, description="Page size (max 50 for this endpoint on many sites)."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params: dict[str, Any] = {"startAt": start_at, "maxResults": max_results}
            if query:
                params["query"] = query
            if expand:
                params["expand"] = expand
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/project/search", params=params)

        super().__init__(handler=_handler, **kwargs)


class JiraGetIssueTypesForProject(Tool):
    name: str = "jira_get_issue_types_for_project"
    description: str | None = "List issue types available for a project."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            project_id_or_key: str = Field(..., description="Project id or key."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                "/rest/api/3/issue/createmeta",
                params={"projectKeys": project_id_or_key, "expand": "projects.issuetypes"},
            )

        super().__init__(handler=_handler, **kwargs)


class JiraCreateIssue(Tool):
    name: str = "jira_create_issue"
    description: str | None = "Create an issue with a Jira fields payload (project, issuetype, summary, etc.)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            fields: dict[str, Any] = Field(
                ...,
                description="Issue fields object, e.g. {'project': {'key': 'ENG'}, 'issuetype': {'name': 'Task'}, 'summary': '...'}.",
            ),
            update_history: bool | None = Field(True, description="Whether to record history for the create."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            body: dict[str, Any] = {"fields": fields}
            params = {}
            if update_history is not None:
                params["updateHistory"] = str(update_history).lower()
            return await _jira_request(
                self, "POST", cloud_id, site_name, "/rest/api/3/issue", params=params or None, json_body=body
            )

        super().__init__(handler=_handler, **kwargs)


class JiraGetIssue(Tool):
    name: str = "jira_get_issue"
    description: str | None = "Get issue details; optional expand for rendered fields, comments, transitions."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            fields: list[str] | None = Field(None, description="If set, only these field ids are returned."),
            expand: str | None = Field(
                "renderedFields,names,schema,operations,editmeta,changelog,versionedRepresentations,transitions",
                description="Comma-separated expand values.",
            ),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params: dict[str, Any] = {}
            if expand:
                params["expand"] = expand
            if fields:
                params["fields"] = ",".join(fields)
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}",
                params=params or None,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraUpdateIssue(Tool):
    name: str = "jira_update_issue"
    description: str | None = "Update issue fields (assignee, priority, status via transition, etc.)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            fields: dict[str, Any] | None = Field(None, description="Fields to set."),
            update: dict[str, Any] | None = Field(None, description="Advanced update document per Jira API."),
            notify_users: bool = Field(True, description="Whether watchers receive notifications."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            body: dict[str, Any] = {}
            if fields:
                body["fields"] = fields
            if update:
                body["update"] = update
            params = {"notifyUsers": str(notify_users).lower()}
            return await _jira_request(
                self,
                "PUT",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}",
                params=params,
                json_body=body or {},
            )

        super().__init__(handler=_handler, **kwargs)


class JiraDeleteIssue(Tool):
    name: str = "jira_delete_issue"
    description: str | None = "Permanently delete an issue."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            delete_subtasks: bool | None = Field(
                None,
                description="When true, child issues may be removed where applicable.",
            ),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params: dict[str, Any] = {}
            if delete_subtasks is not None:
                params["deleteSubtasks"] = str(delete_subtasks).lower()
            return await _jira_request(
                self,
                "DELETE",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}",
                params=params or None,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraTransitionMyIssue(Tool):
    name: str = "jira_transition_my_issue"
    description: str | None = "Move an issue to a new status using a workflow transition id or name."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            transition_id: str | None = Field(None, description="Transition id from GET transitions."),
            transition_name: str | None = Field(None, description="Transition name if id is unknown."),
            fields: dict[str, Any] | None = Field(None, description="Optional fields to set during transition."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            tid = transition_id
            if tid is None and transition_name:
                meta = await _jira_request(
                    self,
                    "GET",
                    cloud_id,
                    site_name,
                    f"/rest/api/3/issue/{issue_id_or_key}/transitions",
                    params={"expand": "transitions.fields"},
                )
                transitions = (meta or {}).get("transitions") or []
                lower = transition_name.strip().lower()
                for t in transitions:
                    if (t.get("name") or "").lower() == lower:
                        tid = str(t.get("id"))
                        break
                if tid is None:
                    raise ValueError(
                        f"No transition named {transition_name!r}. Available: {[t.get('name') for t in transitions]}"
                    )
            if tid is None:
                raise ValueError("Provide transition_id or transition_name.")
            body: dict[str, Any] = {"transition": {"id": tid}}
            if fields:
                body["fields"] = fields
            return await _jira_request(
                self,
                "POST",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}/transitions",
                json_body=body,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraListIssues(Tool):
    name: str = "jira_list_issues"
    description: str | None = "Search issues with JQL (paged)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            jql: str = Field(..., description="JQL query string."),
            start_at: int = Field(0, description="Pagination start."),
            max_results: int = Field(50, description="Page size."),
            fields: list[str] | None = Field(None, description="Field ids to return; default API set if omitted."),
            expand: str | None = Field(None, description="Optional expand string."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            fields_str = ",".join(fields) if fields else None
            return await _jira_search_jql_legacy_shape(
                self,
                cloud_id,
                site_name,
                jql=jql,
                start_at=start_at,
                max_results=max_results,
                fields=fields_str,
                expand=expand,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraExecuteJql(Tool):
    name: str = "jira_execute_jql"
    description: str | None = "Run a raw JQL query against GET /rest/api/3/search/jql (enhanced search)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            jql: str = Field(..., description="Raw JQL."),
            start_at: int = Field(0, description="Pagination start."),
            max_results: int = Field(50, description="Page size."),
            fields: list[str] | None = Field(None, description="Field ids to return."),
            expand: str | None = Field(None, description="Optional expand string."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            fields_str = ",".join(fields) if fields else None
            return await _jira_search_jql_legacy_shape(
                self,
                cloud_id,
                site_name,
                jql=jql,
                start_at=start_at,
                max_results=max_results,
                fields=fields_str,
                expand=expand,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraSearchIssues(Tool):
    name: str = "jira_search_issues"
    description: str | None = (
        "Build JQL from simple filters (project, status, assignee, text) without writing full JQL."
    )
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            project_key: str | None = Field(None, description="Restrict to a project key."),
            status: str | None = Field(None, description="Status name (exact match)."),
            assignee_account_id: str | None = Field(None, description="Assignee account id."),
            reporter_account_id: str | None = Field(None, description="Reporter account id."),
            text: str | None = Field(None, description="Fuzzy text search via summary/description/..."),
            order_by: str = Field("updated DESC", description="ORDER BY clause without the ORDER BY prefix."),
            start_at: int = Field(0, description="Pagination start."),
            max_results: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            parts: list[str] = []
            if project_key:
                parts.append(f'project = "{project_key}"')
            if status:
                parts.append(f'status = "{status}"')
            if assignee_account_id:
                parts.append(f'assignee = "{assignee_account_id}"')
            if reporter_account_id:
                parts.append(f'reporter = "{reporter_account_id}"')
            if text:
                parts.append(f'text ~ "{text}"')
            if parts:
                jql = " AND ".join(parts)
                if order_by.strip():
                    jql = f"{jql} ORDER BY {order_by}"
            else:
                jql = (
                    f"issuetype is not EMPTY ORDER BY {order_by}"
                    if order_by.strip()
                    else "issuetype is not EMPTY ORDER BY updated DESC"
                )
            return await _jira_search_jql_legacy_shape(
                self,
                cloud_id,
                site_name,
                jql=jql,
                start_at=start_at,
                max_results=max_results,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraCommentOnIssue(Tool):
    name: str = "jira_comment_on_issue"
    description: str | None = "Add a comment to an issue (Atlassian Document Format body supported)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            body_adf: dict[str, Any] = Field(
                ...,
                description="Comment body in Atlassian Document Format, e.g. {'type':'doc','version':1,'content':[{'type':'paragraph','content':[{'type':'text','text':'Hello'}]}]}.",
            ),
            visibility: dict[str, Any] | None = Field(None, description="Optional visibility restriction."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            payload: dict[str, Any] = {"body": body_adf}
            if visibility:
                payload["visibility"] = visibility
            return await _jira_request(
                self,
                "POST",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}/comment",
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraAddAttachment(Tool):
    name: str = "jira_add_attachment"
    description: str | None = "Attach a file to an issue using base64 file content."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            filename: str = Field(..., description="Filename for the attachment."),
            file_base64: str = Field(..., description="File bytes encoded as standard base64."),
            content_type: str | None = Field(None, description="Optional MIME type."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            raw = base64.b64decode(file_base64)
            base, hdrs = await _jira_connection(self, cloud_id, site_name)
            url = f"{base}/rest/api/3/issue/{issue_id_or_key}/attachments"
            headers = {**hdrs, "X-Atlassian-Token": "no-check"}
            import httpx

            ct = content_type or "application/octet-stream"
            files = {"file": (filename, raw, ct)}
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, files=files)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_handler, **kwargs)


class JiraListFields(Tool):
    name: str = "jira_list_fields"
    description: str | None = "List all fields including custom fields."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/field")

        super().__init__(handler=_handler, **kwargs)


class JiraGetEditMetadata(Tool):
    name: str = "jira_get_edit_metadata"
    description: str | None = "Get editable fields and allowed values for an issue."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}/editmeta",
            )

        super().__init__(handler=_handler, **kwargs)


class JiraAddUserToIssue(Tool):
    name: str = "jira_add_user_to_issue"
    description: str | None = "Set assignee or reporter, or add a watcher, using an account id."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            account_id: str = Field(..., description="Atlassian account id of the user."),
            role: str = Field(
                ...,
                description="Whether to set assignee, reporter, or add watcher.",
                json_schema_extra={"enum": ["assignee", "reporter", "watcher"]},
            ),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            if role not in ("assignee", "reporter", "watcher"):
                raise ValueError(f"role must be assignee, reporter, or watcher; got {role!r}")
            if role == "watcher":
                base, hdrs = await _jira_connection(self, cloud_id, site_name)
                url = f"{base}/rest/api/3/issue/{issue_id_or_key}/watchers"
                headers = {**hdrs, "Content-Type": "application/json"}
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, headers=headers, content=json.dumps(account_id))
                    response.raise_for_status()
                    if response.status_code == 204 or not response.content:
                        return None
                    return response.json()
            if role == "assignee":
                return await _jira_request(
                    self,
                    "PUT",
                    cloud_id,
                    site_name,
                    f"/rest/api/3/issue/{issue_id_or_key}/assignee",
                    json_body={"accountId": account_id},
                )
            return await _jira_request(
                self,
                "PUT",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}",
                json_body={"fields": {"reporter": {"accountId": account_id}}},
            )

        super().__init__(handler=_handler, **kwargs)


class JiraGetMyself(Tool):
    name: str = "jira_get_myself"
    description: str | None = "Return the authenticated user's profile."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/myself")

        super().__init__(handler=_handler, **kwargs)


class JiraGetMyIssues(Tool):
    name: str = "jira_get_my_issues"
    description: str | None = "List open issues assigned to the current user."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            start_at: int = Field(0, description="Pagination start."),
            max_results: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            jql = "assignee = currentUser() AND resolution = Unresolved ORDER BY updated DESC"
            return await _jira_search_jql_legacy_shape(
                self,
                cloud_id,
                site_name,
                jql=jql,
                start_at=start_at,
                max_results=max_results,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraGetMyRecentActivity(Tool):
    name: str = "jira_get_my_recent_activity"
    description: str | None = "Recently updated issues the current user reported, is assigned to, or watches."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            max_results: int = Field(20, description="How many issues to return."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            jql = (
                "(assignee = currentUser() OR reporter = currentUser() OR watcher = currentUser()) "
                "ORDER BY updated DESC"
            )
            return await _jira_search_jql_legacy_shape(
                self,
                cloud_id,
                site_name,
                jql=jql,
                start_at=0,
                max_results=max_results,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraGetMyPermissions(Tool):
    name: str = "jira_get_my_permissions"
    description: str | None = "Resolve permissions for the current user in a project or globally."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            permissions: str = Field(
                "BROWSE,EDIT_ISSUES,CREATE_ISSUES,TRANSITION_ISSUES",
                description="Comma-separated permission keys.",
            ),
            project_key: str | None = Field(None, description="Project key context."),
            issue_key: str | None = Field(None, description="Issue key context."),
            project_id: str | None = Field(None, description="Numeric project id context."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params: dict[str, Any] = {"permissions": permissions}
            if project_key:
                params["projectKey"] = project_key
            if issue_key:
                params["issueKey"] = issue_key
            if project_id:
                params["projectId"] = project_id
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/mypermissions", params=params)

        super().__init__(handler=_handler, **kwargs)


class JiraListUsers(Tool):
    name: str = "jira_list_users"
    description: str | None = "Search users readable by the current token (query required by Jira Cloud)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            query: str = Field(
                "@",
                description="Search query; '@' is a common prefix to list many users your token may browse.",
            ),
            start_at: int = Field(0, description="Pagination start."),
            max_results: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {"query": query, "startAt": start_at, "maxResults": max_results}
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/user/search", params=params)

        super().__init__(handler=_handler, **kwargs)


class JiraListGroups(Tool):
    name: str = "jira_list_groups"
    description: str | None = "Bulk list groups (requires browse permission to group picker APIs)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            start_at: int = Field(0, description="Pagination start."),
            max_results: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {"startAt": start_at, "maxResults": max_results}
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/group/bulk", params=params)

        super().__init__(handler=_handler, **kwargs)


class JiraCreateGroup(Tool):
    name: str = "jira_create_group"
    description: str | None = "Create a new group (admin capability on many sites)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            name: str = Field(..., description="Group name."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(self, "POST", cloud_id, site_name, "/rest/api/3/group", json_body={"name": name})

        super().__init__(handler=_handler, **kwargs)


class JiraAddUserToGroup(Tool):
    name: str = "jira_add_user_to_group"
    description: str | None = "Add a user (account id) to a named group."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            group_name: str = Field(..., description="Group name."),
            account_id: str = Field(..., description="User account id."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {"groupname": group_name}
            return await _jira_request(
                self,
                "POST",
                cloud_id,
                site_name,
                "/rest/api/3/group/user",
                params=params,
                json_body={"accountId": account_id},
            )

        super().__init__(handler=_handler, **kwargs)


class JiraRemoveUserFromGroup(Tool):
    name: str = "jira_remove_user_from_group"
    description: str | None = "Remove a user from a named group."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            group_name: str = Field(..., description="Group name."),
            account_id: str = Field(..., description="User account id."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {"groupname": group_name, "accountId": account_id}
            return await _jira_request(
                self,
                "DELETE",
                cloud_id,
                site_name,
                "/rest/api/3/group/user",
                params=params,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraListIssueLinkTypes(Tool):
    name: str = "jira_list_issue_link_types"
    description: str | None = "List issue link types (Duplicate, Blocks, ...)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/api/3/issueLinkType")

        super().__init__(handler=_handler, **kwargs)


class JiraCreateIssueLink(Tool):
    name: str = "jira_create_issue_link"
    description: str | None = "Link two issues (inward/outward keys depend on link type)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            link_type_name: str = Field(..., description="Name of the link type."),
            inward_issue_key: str = Field(..., description="Inward issue key."),
            outward_issue_key: str = Field(..., description="Outward issue key."),
            comment: dict[str, Any] | None = Field(None, description="Optional comment object."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            body: dict[str, Any] = {
                "type": {"name": link_type_name},
                "inwardIssue": {"key": inward_issue_key},
                "outwardIssue": {"key": outward_issue_key},
            }
            if comment:
                body["comment"] = comment
            return await _jira_request(self, "POST", cloud_id, site_name, "/rest/api/3/issueLink", json_body=body)

        super().__init__(handler=_handler, **kwargs)


class JiraDeleteIssueLink(Tool):
    name: str = "jira_delete_issue_link"
    description: str | None = "Delete an issue link by id."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            link_id: str = Field(..., description="Issue link id."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(self, "DELETE", cloud_id, site_name, f"/rest/api/3/issueLink/{link_id}")

        super().__init__(handler=_handler, **kwargs)


class JiraGetIssueLinks(Tool):
    name: str = "jira_get_issue_links"
    description: str | None = "Return issuelinks for an issue."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Issue id or key."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/api/3/issue/{issue_id_or_key}",
                params={"fields": "issuelinks"},
            )

        super().__init__(handler=_handler, **kwargs)


class JiraListServiceDesks(Tool):
    name: str = "jira_list_service_desks"
    description: str | None = "List Jira Service Management service desks."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            start: int = Field(0, description="Pagination start."),
            limit: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {"start": start, "limit": limit}
            return await _jira_request(
                self, "GET", cloud_id, site_name, "/rest/servicedeskapi/servicedesk", params=params
            )

        super().__init__(handler=_handler, **kwargs)


class JiraListRequestTypes(Tool):
    name: str = "jira_list_request_types"
    description: str | None = "List request types for a service desk."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            service_desk_id: str = Field(..., description="Service desk id."),
            start: int = Field(0, description="Pagination start."),
            limit: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {"start": start, "limit": limit}
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/servicedeskapi/servicedesk/{service_desk_id}/requesttype",
                params=params,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraGetRequestTypeFields(Tool):
    name: str = "jira_get_request_type_fields"
    description: str | None = "Get fields required to raise a request of a given type."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            service_desk_id: str = Field(..., description="Service desk id."),
            request_type_id: str = Field(..., description="Request type id."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/servicedeskapi/servicedesk/{service_desk_id}/requesttype/{request_type_id}/field",
            )

        super().__init__(handler=_handler, **kwargs)


class JiraCreateCustomerRequest(Tool):
    name: str = "jira_create_customer_request"
    description: str | None = "Create a customer request on a service desk (use get_request_type_fields first)."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            service_desk_id: str = Field(..., description="Service desk id."),
            request_type_id: str = Field(..., description="Request type id."),
            request_field_values: dict[str, Any] = Field(
                ...,
                description="Field values map, e.g. {'summary': '...', 'description': '...'}.",
            ),
            raise_on_behalf_of: str | None = Field(
                None,
                description="Optional customer account id to create on behalf of.",
            ),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            body: dict[str, Any] = {
                "serviceDeskId": service_desk_id,
                "requestTypeId": request_type_id,
                "requestFieldValues": request_field_values,
            }
            if raise_on_behalf_of:
                body["raiseOnBehalfOf"] = raise_on_behalf_of
            return await _jira_request(
                self, "POST", cloud_id, site_name, "/rest/servicedeskapi/request", json_body=body
            )

        super().__init__(handler=_handler, **kwargs)


class JiraGetCustomerRequest(Tool):
    name: str = "jira_get_customer_request"
    description: str | None = "Get a customer request by issue key."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            issue_id_or_key: str = Field(..., description="Customer request issue key."),
            expand: str | None = Field(None, description="Optional expand per servicedesk API."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params = {}
            if expand:
                params["expand"] = expand
            return await _jira_request(
                self,
                "GET",
                cloud_id,
                site_name,
                f"/rest/servicedeskapi/request/{issue_id_or_key}",
                params=params or None,
            )

        super().__init__(handler=_handler, **kwargs)


class JiraListCustomerRequests(Tool):
    name: str = "jira_list_customer_requests"
    description: str | None = "List customer requests with optional filters."
    integration: Annotated[str, Integration("jira")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            search_query: str | None = Field(None, description="Search string."),
            service_desk_id: str | None = Field(None, description="Filter by service desk id."),
            request_status: str | None = Field(None, description="OPEN, CLOSED, etc. when supported."),
            start: int = Field(0, description="Pagination start."),
            limit: int = Field(50, description="Page size."),
            cloud_id: str | None = Field(None, description="Atlassian cloud id when multiple sites exist."),
            site_name: str | None = Field(None, description="Site selector when multiple sites exist."),
        ) -> Any:
            params: dict[str, Any] = {"start": start, "limit": limit}
            if search_query:
                params["searchQuery"] = search_query
            if service_desk_id:
                params["serviceDeskId"] = service_desk_id
            if request_status:
                params["requestStatus"] = request_status
            return await _jira_request(self, "GET", cloud_id, site_name, "/rest/servicedeskapi/request", params=params)

        super().__init__(handler=_handler, **kwargs)


_JIRA_TOOL_CLASSES: tuple[type[Tool], ...] = (
    JiraCreateProject,
    JiraGetProject,
    JiraUpdateProject,
    JiraDeleteProject,
    JiraListProjects,
    JiraGetIssueTypesForProject,
    JiraCreateIssue,
    JiraGetIssue,
    JiraUpdateIssue,
    JiraDeleteIssue,
    JiraTransitionMyIssue,
    JiraListIssues,
    JiraExecuteJql,
    JiraSearchIssues,
    JiraCommentOnIssue,
    JiraAddAttachment,
    JiraListFields,
    JiraGetEditMetadata,
    JiraAddUserToIssue,
    JiraGetMyself,
    JiraGetMyIssues,
    JiraGetMyRecentActivity,
    JiraGetMyPermissions,
    JiraListUsers,
    JiraListGroups,
    JiraCreateGroup,
    JiraAddUserToGroup,
    JiraRemoveUserFromGroup,
    JiraListIssueLinkTypes,
    JiraCreateIssueLink,
    JiraDeleteIssueLink,
    JiraGetIssueLinks,
    JiraListServiceDesks,
    JiraListRequestTypes,
    JiraGetRequestTypeFields,
    JiraCreateCustomerRequest,
    JiraGetCustomerRequest,
    JiraListCustomerRequests,
)

JIRA_TOOL_INSTANCES: list[Tool] = [cls() for cls in _JIRA_TOOL_CLASSES]



async def main():
    # get_issue = JiraGetIssue()
    list_projects = JiraListProjects()
    result = await list_projects().collect()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())