import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_ASANA_API_BASE = "https://app.asana.com/api/1.0"


async def _resolve_token(tool: Any) -> str:
    """Resolve Asana access token from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials.get("token") or credentials["api_key"]
    if getattr(tool, "token", None) is not None:
        return tool.token.get_secret_value()
    env_token = os.getenv("ASANA_ACCESS_TOKEN")
    if env_token:
        return env_token
    raise ValueError(
        "Asana access token not found. Set ASANA_ACCESS_TOKEN, pass token, or configure an integration."
    )


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
    }


class AsanaCreateProject(Tool):
    name: str = "asana_create_project"
    description: str | None = "Create a new Asana project in a workspace or team."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_project(
            name: str = Field(..., description="Name of the project to create."),
            workspace_gid: str = Field(..., description="Workspace or organization GID where the project will be created."),
            team_gid: str | None = Field(
                None,
                description="Team GID to share the project with (required when the workspace is an organization).",
            ),
            notes: str | None = Field(None, description="Optional description for the project."),
            due_on: str | None = Field(
                None, description="Optional due date for the project in YYYY-MM-DD format."
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {
                "name": name,
                "workspace": workspace_gid,
            }
            if team_gid:
                data["team"] = team_gid
            if notes:
                data["notes"] = notes
            if due_on:
                data["due_on"] = due_on

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ASANA_API_BASE}/projects",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_project, **kwargs)


class AsanaListWorkspaces(Tool):
    name: str = "asana_list_workspaces"
    description: str | None = "List Asana workspaces and organizations accessible with the current token."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspaces() -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/workspaces",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_workspaces, **kwargs)


class AsanaUpdateTask(Tool):
    name: str = "asana_update_task"
    description: str | None = "Update an existing Asana task."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_task(
            task_gid: str = Field(..., description="GID of the task to update."),
            name: str | None = Field(None, description="Updated task name."),
            notes: str | None = Field(None, description="Updated task notes / description."),
            completed: bool | None = Field(None, description="Whether the task is completed."),
            due_on: str | None = Field(None, description="Due date for the task in YYYY-MM-DD format."),
            assignee_gid: str | None = Field(None, description="User GID to assign the task to."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {}
            if name is not None:
                data["name"] = name
            if notes is not None:
                data["notes"] = notes
            if completed is not None:
                data["completed"] = completed
            if due_on is not None:
                data["due_on"] = due_on
            if assignee_gid is not None:
                data["assignee"] = assignee_gid

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_ASANA_API_BASE}/tasks/{task_gid}",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_task, **kwargs)


class AsanaListUserProjects(Tool):
    name: str = "asana_list_user_projects"
    description: str | None = "List Asana projects for a given user in a workspace."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_user_projects(
            workspace_gid: str = Field(..., description="Workspace or organization GID."),
            owner_gid: str | None = Field(
                None, description="User GID whose projects to list. If omitted, returns all accessible projects."
            ),
            archived: bool = Field(False, description="Whether to include archived projects."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "workspace": workspace_gid,
                "archived": str(archived).lower(),
            }
            if owner_gid:
                params["owner"] = owner_gid

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/projects",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_user_projects, **kwargs)


class AsanaListTeams(Tool):
    name: str = "asana_list_teams"
    description: str | None = "List teams for a given organization (workspace) GID."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_teams(
            organization_gid: str = Field(
                ...,
                description="Organization (workspace) GID whose teams should be listed.",
            ),
            limit: int = Field(50, description="Maximum number of teams to return."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "organization": organization_gid,
                "limit": limit,
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/teams",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_teams, **kwargs)


class AsanaSearchTasks(Tool):
    name: str = "asana_search_tasks"
    description: str | None = "Search tasks by name within a project."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_tasks(
            workspace_gid: str = Field(..., description="Workspace or organization GID for the search."),
            project_gid: str = Field(..., description="Project GID to search within."),
            text: str = Field(..., description="Search text to match against task names."),
            limit: int = Field(50, description="Maximum number of tasks to return."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "workspace": workspace_gid,
                "project": project_gid,
                "text": text,
                "limit": limit,
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/tasks/search",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_tasks, **kwargs)


class AsanaSearchSections(Tool):
    name: str = "asana_search_sections"
    description: str | None = "Search sections by name within a project."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_sections(
            project_gid: str = Field(..., description="Project GID whose sections will be searched."),
            name_query: str = Field(..., description="Section name to search for."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/projects/{project_gid}/sections",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                sections = response.json().get("data", [])
                filtered = [s for s in sections if name_query.lower() in s.get("name", "").lower()]
                return {"data": filtered}

        super().__init__(handler=_search_sections, **kwargs)


class AsanaSearchProjects(Tool):
    name: str = "asana_search_projects"
    description: str | None = "Search projects by name within a workspace."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_projects(
            workspace_gid: str = Field(..., description="Workspace or organization GID."),
            name_query: str = Field(..., description="Project name to search for."),
            archived: bool = Field(False, description="Whether to include archived projects."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "workspace": workspace_gid,
                "archived": str(archived).lower(),
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/projects",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                projects = response.json().get("data", [])
                filtered = [p for p in projects if name_query.lower() in p.get("name", "").lower()]
                return {"data": filtered}

        super().__init__(handler=_search_projects, **kwargs)


class AsanaListTaskStories(Tool):
    name: str = "asana_list_task_stories"
    description: str | None = "List stories (including comments) for a task."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_task_stories(
            task_gid: str = Field(..., description="Task GID whose stories to list."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/tasks/{task_gid}/stories",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_task_stories, **kwargs)


class AsanaGetTasksFromTaskList(Tool):
    name: str = "asana_get_tasks_from_task_list"
    description: str | None = "Get tasks from a user's My Tasks list (user task list)."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_tasks_from_task_list(
            user_task_list_gid: str | None = Field(
                None,
                description="User task list (My Tasks) GID. Provide this OR user_gid.",
            ),
            user_gid: str | None = Field(
                None,
                description="User GID whose My Tasks list should be used when user_task_list_gid is not known.",
            ),
            limit: int = Field(50, description="Maximum number of tasks to return."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            if not user_task_list_gid and not user_gid:
                raise ValueError("You must provide either user_task_list_gid or user_gid.")

            resolved_user_task_list_gid = user_task_list_gid

            async with httpx.AsyncClient() as client:
                # If only user_gid is provided, resolve the corresponding user_task_list first.
                if resolved_user_task_list_gid is None and user_gid is not None:
                    resolve_resp = await client.get(
                        f"{_ASANA_API_BASE}/users/{user_gid}/user_task_list",
                        headers=_auth_headers(token),
                    )
                    resolve_resp.raise_for_status()
                    resolved_user_task_list_gid = resolve_resp.json()["data"]["gid"]

                params: dict[str, Any] = {
                    "limit": limit,
                }

                response = await client.get(
                    f"{_ASANA_API_BASE}/user_task_lists/{resolved_user_task_list_gid}/tasks",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_tasks_from_task_list, **kwargs)


class AsanaFindTaskById(Tool):
    name: str = "asana_find_task_by_id"
    description: str | None = "Find a task by its GID and return the complete record."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_task_by_id(
            task_gid: str = Field(..., description="GID of the task to fetch."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ASANA_API_BASE}/tasks/{task_gid}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_find_task_by_id, **kwargs)


class AsanaDeleteTask(Tool):
    name: str = "asana_delete_task"
    description: str | None = "Delete a specific Asana task."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_task(
            task_gid: str = Field(..., description="GID of the task to delete."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_ASANA_API_BASE}/tasks/{task_gid}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json() if response.content else {"success": True}

        super().__init__(handler=_delete_task, **kwargs)


class AsanaCreateTask(Tool):
    name: str = "asana_create_task"
    description: str | None = "Create a new Asana task."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task(
            workspace_gid: str = Field(..., description="Workspace or organization GID."),
            name: str = Field(..., description="Task name."),
            notes: str | None = Field(None, description="Task notes / description."),
            assignee_gid: str | None = Field(None, description="User GID to assign the task to."),
            projects: list[str] | None = Field(None, description="List of project GIDs to add the task to."),
            parent_task_gid: str | None = Field(None, description="Parent task GID if creating a subtask."),
            due_on: str | None = Field(None, description="Due date for the task in YYYY-MM-DD format."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {
                "workspace": workspace_gid,
                "name": name,
            }
            if notes is not None:
                data["notes"] = notes
            if assignee_gid is not None:
                data["assignee"] = assignee_gid
            if projects:
                data["projects"] = projects
            if parent_task_gid is not None:
                data["parent"] = parent_task_gid
            if due_on is not None:
                data["due_on"] = due_on

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ASANA_API_BASE}/tasks",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_task, **kwargs)


class AsanaCreateTaskFromTemplate(Tool):
    name: str = "asana_create_task_from_template"
    description: str | None = "Create a new Asana task from a task template."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task_from_template(
            task_template_gid: str = Field(..., description="Task template GID to use."),
            name: str | None = Field(None, description="Optional override for task name."),
            assignee_gid: str | None = Field(None, description="User GID to assign the task to."),
            workspace_gid: str | None = Field(
                None, description="Workspace or organization GID. Required if not implied by the template."
            ),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {
                "task_template": task_template_gid,
            }
            if name is not None:
                data["name"] = name
            if assignee_gid is not None:
                data["assignee"] = assignee_gid
            if workspace_gid is not None:
                data["workspace"] = workspace_gid

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ASANA_API_BASE}/tasks",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_task_from_template, **kwargs)


class AsanaCreateTaskComment(Tool):
    name: str = "asana_create_task_comment"
    description: str | None = "Add a comment to an Asana task."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task_comment(
            task_gid: str = Field(..., description="GID of the task to comment on."),
            text: str = Field(..., description="Comment text."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data = {
                "text": text,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ASANA_API_BASE}/tasks/{task_gid}/stories",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_task_comment, **kwargs)


class AsanaCreateSubtask(Tool):
    name: str = "asana_create_subtask"
    description: str | None = "Create a new subtask under a parent task."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_subtask(
            parent_task_gid: str = Field(..., description="Parent task GID."),
            name: str = Field(..., description="Subtask name."),
            notes: str | None = Field(None, description="Subtask notes / description."),
            assignee_gid: str | None = Field(None, description="User GID to assign the subtask to."),
            due_on: str | None = Field(None, description="Due date for the subtask in YYYY-MM-DD format."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {
                "name": name,
            }
            if notes is not None:
                data["notes"] = notes
            if assignee_gid is not None:
                data["assignee"] = assignee_gid
            if due_on is not None:
                data["due_on"] = due_on

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ASANA_API_BASE}/tasks/{parent_task_gid}/subtasks",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_subtask, **kwargs)


class AsanaAddTaskToSection(Tool):
    name: str = "asana_add_task_to_section"
    description: str | None = "Add a task to a specific section in a project."
    integration: Annotated[str, Integration("asana")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_task_to_section(
            section_gid: str = Field(..., description="Section GID to add the task to."),
            task_gid: str = Field(..., description="Task GID to move into the section."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data = {
                "task": task_gid,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ASANA_API_BASE}/sections/{section_gid}/addTask",
                    headers=_auth_headers(token),
                    json={"data": data},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_add_task_to_section, **kwargs)

