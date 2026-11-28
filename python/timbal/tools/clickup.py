import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_CLICKUP_API_BASE = "https://api.clickup.com/api/v2"


async def _resolve_token(tool: Any) -> str:
    """Resolve a ClickUp token from the integration, an explicit field, or an env var.

    ClickUp accepts both a personal API token and an OAuth access token the same
    way: verbatim in the `Authorization` header, with no `Bearer` prefix.
    """
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials.get("access_token") or credentials["token"]
    if getattr(tool, "token", None) is not None:
        return tool.token.get_secret_value()
    env_token = os.getenv("CLICKUP_API_TOKEN")
    if env_token:
        return env_token
    raise ValueError(
        "ClickUp token not found. Set CLICKUP_API_TOKEN, pass token, or configure an integration."
    )


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": token,
        "Content-Type": "application/json",
    }


class ClickUpListWorkspaces(Tool):
    name: str = "clickup_list_workspaces"
    description: str | None = "List the ClickUp workspaces (teams) the current token can access."
    integration: Annotated[str, Integration("clickup")] | None = None
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

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/team",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_workspaces, **kwargs)


class ClickUpListSpaces(Tool):
    name: str = "clickup_list_spaces"
    description: str | None = "List the spaces in a ClickUp workspace."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_spaces(
            workspace_id: str = Field(..., description="Workspace (team) ID whose spaces to list."),
            archived: bool = Field(False, description="Whether to include archived spaces."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/team/{workspace_id}/space",
                    headers=_auth_headers(token),
                    params={"archived": str(archived).lower()},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_spaces, **kwargs)


class ClickUpListFolders(Tool):
    name: str = "clickup_list_folders"
    description: str | None = "List the folders in a ClickUp space."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_folders(
            space_id: str = Field(..., description="Space ID whose folders to list."),
            archived: bool = Field(False, description="Whether to include archived folders."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/space/{space_id}/folder",
                    headers=_auth_headers(token),
                    params={"archived": str(archived).lower()},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_folders, **kwargs)


class ClickUpCreateFolder(Tool):
    name: str = "clickup_create_folder"
    description: str | None = "Create a new folder in a ClickUp space."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_folder(
            space_id: str = Field(..., description="Space ID the folder will be created in."),
            name: str = Field(..., description="Name of the new folder."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CLICKUP_API_BASE}/space/{space_id}/folder",
                    headers=_auth_headers(token),
                    json={"name": name},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_folder, **kwargs)


class ClickUpListLists(Tool):
    name: str = "clickup_list_lists"
    description: str | None = "List the lists inside a ClickUp folder."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_lists(
            folder_id: str = Field(..., description="Folder ID whose lists to list."),
            archived: bool = Field(False, description="Whether to include archived lists."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/folder/{folder_id}/list",
                    headers=_auth_headers(token),
                    params={"archived": str(archived).lower()},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_lists, **kwargs)


class ClickUpListFolderlessLists(Tool):
    name: str = "clickup_list_folderless_lists"
    description: str | None = "List the lists that live directly in a ClickUp space, outside any folder."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_folderless_lists(
            space_id: str = Field(..., description="Space ID whose folderless lists to list."),
            archived: bool = Field(False, description="Whether to include archived lists."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/space/{space_id}/list",
                    headers=_auth_headers(token),
                    params={"archived": str(archived).lower()},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_folderless_lists, **kwargs)


class ClickUpCreateList(Tool):
    name: str = "clickup_create_list"
    description: str | None = "Create a new list inside a ClickUp folder."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_list(
            folder_id: str = Field(..., description="Folder ID the list will be created in."),
            name: str = Field(..., description="Name of the new list."),
            content: str | None = Field(None, description="Optional description for the list."),
            due_date: int | None = Field(None, description="Optional due date for the list, as Unix ms."),
            priority: int | None = Field(None, description="Optional priority: 1 urgent, 2 high, 3 normal, 4 low."),
            assignee: int | None = Field(None, description="Optional user ID to assign the list to."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {"name": name}
            if content is not None:
                data["content"] = content
            if due_date is not None:
                data["due_date"] = due_date
            if priority is not None:
                data["priority"] = priority
            if assignee is not None:
                data["assignee"] = assignee

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CLICKUP_API_BASE}/folder/{folder_id}/list",
                    headers=_auth_headers(token),
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_list, **kwargs)


class ClickUpGetTask(Tool):
    name: str = "clickup_get_task"
    description: str | None = "Get a ClickUp task by its ID."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_task(
            task_id: str = Field(..., description="ID of the task to fetch."),
            include_subtasks: bool = Field(False, description="Whether to include the task's subtasks."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/task/{task_id}",
                    headers=_auth_headers(token),
                    params={"include_subtasks": str(include_subtasks).lower()},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_task, **kwargs)


class ClickUpListTasks(Tool):
    name: str = "clickup_list_tasks"
    description: str | None = "List the tasks in a ClickUp list, optionally filtered by status or assignee."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tasks(
            list_id: str = Field(..., description="List ID whose tasks to list."),
            archived: bool = Field(False, description="Whether to include archived tasks."),
            include_closed: bool = Field(False, description="Whether to include closed tasks."),
            statuses: list[str] | None = Field(None, description="Only return tasks with one of these statuses."),
            assignees: list[str] | None = Field(None, description="Only return tasks assigned to one of these user IDs."),
            page: int = Field(0, description="Zero-based page number for pagination."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "archived": str(archived).lower(),
                "include_closed": str(include_closed).lower(),
                "page": page,
            }
            if statuses:
                params["statuses[]"] = statuses
            if assignees:
                params["assignees[]"] = assignees

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/list/{list_id}/task",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_tasks, **kwargs)


class ClickUpSearchTasks(Tool):
    name: str = "clickup_search_tasks"
    description: str | None = (
        "Search for tasks across a whole ClickUp workspace, filtered by space, folder, list, "
        "status, assignee or tag."
    )
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_tasks(
            workspace_id: str = Field(..., description="Workspace (team) ID to search within."),
            space_ids: list[str] | None = Field(None, description="Only search within these space IDs."),
            list_ids: list[str] | None = Field(None, description="Only search within these list IDs."),
            statuses: list[str] | None = Field(None, description="Only return tasks with one of these statuses."),
            assignees: list[str] | None = Field(None, description="Only return tasks assigned to one of these user IDs."),
            tags: list[str] | None = Field(None, description="Only return tasks carrying one of these tag names."),
            include_closed: bool = Field(False, description="Whether to include closed tasks."),
            page: int = Field(0, description="Zero-based page number for pagination."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "include_closed": str(include_closed).lower(),
                "page": page,
            }
            if space_ids:
                params["space_ids[]"] = space_ids
            if list_ids:
                params["list_ids[]"] = list_ids
            if statuses:
                params["statuses[]"] = statuses
            if assignees:
                params["assignees[]"] = assignees
            if tags:
                params["tags[]"] = tags

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/team/{workspace_id}/task",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_tasks, **kwargs)


class ClickUpCreateTask(Tool):
    name: str = "clickup_create_task"
    description: str | None = "Create a new task in a ClickUp list."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task(
            list_id: str = Field(..., description="List ID the task will be created in."),
            name: str = Field(..., description="Task name."),
            description: str | None = Field(None, description="Task description."),
            status: str | None = Field(None, description="Status name to create the task with."),
            priority: int | None = Field(None, description="Priority: 1 urgent, 2 high, 3 normal, 4 low."),
            assignees: list[int] | None = Field(None, description="User IDs to assign the task to."),
            tags: list[str] | None = Field(None, description="Tag names to attach to the task."),
            due_date: int | None = Field(None, description="Due date as Unix ms."),
            parent: str | None = Field(None, description="Parent task ID, to create this task as a subtask."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {"name": name}
            if description is not None:
                data["description"] = description
            if status is not None:
                data["status"] = status
            if priority is not None:
                data["priority"] = priority
            if assignees:
                data["assignees"] = assignees
            if tags:
                data["tags"] = tags
            if due_date is not None:
                data["due_date"] = due_date
            if parent is not None:
                data["parent"] = parent

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CLICKUP_API_BASE}/list/{list_id}/task",
                    headers=_auth_headers(token),
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_task, **kwargs)


class ClickUpUpdateTask(Tool):
    name: str = "clickup_update_task"
    description: str | None = "Update an existing ClickUp task."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_task(
            task_id: str = Field(..., description="ID of the task to update."),
            name: str | None = Field(None, description="Updated task name."),
            description: str | None = Field(None, description="Updated task description."),
            status: str | None = Field(None, description="Updated status name."),
            priority: int | None = Field(None, description="Updated priority: 1 urgent, 2 high, 3 normal, 4 low."),
            due_date: int | None = Field(None, description="Updated due date as Unix ms."),
            archived: bool | None = Field(None, description="Whether the task should be archived."),
            add_assignees: list[int] | None = Field(None, description="User IDs to add as assignees."),
            remove_assignees: list[int] | None = Field(None, description="User IDs to remove as assignees."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {}
            if name is not None:
                data["name"] = name
            if description is not None:
                data["description"] = description
            if status is not None:
                data["status"] = status
            if priority is not None:
                data["priority"] = priority
            if due_date is not None:
                data["due_date"] = due_date
            if archived is not None:
                data["archived"] = archived
            if add_assignees or remove_assignees:
                data["assignees"] = {
                    "add": add_assignees or [],
                    "rem": remove_assignees or [],
                }

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.put(
                    f"{_CLICKUP_API_BASE}/task/{task_id}",
                    headers=_auth_headers(token),
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_task, **kwargs)


class ClickUpDeleteTask(Tool):
    name: str = "clickup_delete_task"
    description: str | None = "Delete a ClickUp task."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_task(
            task_id: str = Field(..., description="ID of the task to delete."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.delete(
                    f"{_CLICKUP_API_BASE}/task/{task_id}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json() if response.content else {"success": True}

        super().__init__(handler=_delete_task, **kwargs)


class ClickUpCreateTaskComment(Tool):
    name: str = "clickup_create_task_comment"
    description: str | None = "Add a comment to a ClickUp task."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task_comment(
            task_id: str = Field(..., description="ID of the task to comment on."),
            comment_text: str = Field(..., description="Comment text."),
            assignee: int | None = Field(None, description="Optional user ID to assign the comment to."),
            notify_all: bool = Field(True, description="Whether to notify everyone on the task."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            data: dict[str, Any] = {
                "comment_text": comment_text,
                "notify_all": notify_all,
            }
            if assignee is not None:
                data["assignee"] = assignee

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CLICKUP_API_BASE}/task/{task_id}/comment",
                    headers=_auth_headers(token),
                    json=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_task_comment, **kwargs)


class ClickUpGetTaskComments(Tool):
    name: str = "clickup_get_task_comments"
    description: str | None = "List the comments on a ClickUp task."
    integration: Annotated[str, Integration("clickup")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_task_comments(
            task_id: str = Field(..., description="ID of the task whose comments to list."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CLICKUP_API_BASE}/task/{task_id}/comment",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_task_comments, **kwargs)
