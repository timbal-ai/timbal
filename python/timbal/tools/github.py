from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_GITHUB_BASE = "https://api.github.com"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        token = credentials.get("token")
        if token:
            return token
        raise ValueError("Integration credentials did not include a token.")
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError(
        "GitHub credentials not found. Configure an integration or pass token."
    )


def _repo(owner: str, repo: str) -> str:
    return f"{_GITHUB_BASE}/repos/{owner}/{repo}"


# ---------------------------------------------------------------------------
# Git Data — References & Commits
# ---------------------------------------------------------------------------


class GetReference(Tool):
    name: str = "github_get_reference"
    description: str | None = "Get a single git reference (branch, tag, or HEAD) from a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_reference(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            ref: str = Field(..., description="Fully-qualified reference, e.g. 'heads/main' or 'tags/v1.0'")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/git/ref/{ref}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_reference, **kwargs)


class CreateReference(Tool):
    name: str = "github_create_reference"
    description: str | None = "Create a git reference (branch or tag) in a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_reference(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            ref: str = Field(..., description="Fully-qualified reference name, e.g. 'refs/heads/my-branch'"),
            sha: str = Field(..., description="The SHA1 value to set the reference to"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_repo(owner, repo)}/git/refs",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ref": ref, "sha": sha},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CreateReference"
        super().__init__(handler=_create_reference, **kwargs)


class UpdateReference(Tool):
    name: str = "github_update_reference"
    description: str | None = "Update a git reference to point to a new commit SHA."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_reference(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            ref: str = Field(..., description="Reference to update, e.g. 'heads/main'"),
            sha: str = Field(..., description="New commit SHA to point the reference to"),
            force: bool = Field(False, description="If True, force-update even if not a fast-forward"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_repo(owner, repo)}/git/refs/{ref}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"sha": sha, "force": force},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/UpdateReference"
        super().__init__(handler=_update_reference, **kwargs)


class DeleteReference(Tool):
    name: str = "github_delete_reference"
    description: str | None = "Delete a git reference (branch or tag) from a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_reference(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            ref: str = Field(..., description="Reference to delete, e.g. 'heads/my-branch'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_repo(owner, repo)}/git/refs/{ref}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "ref": ref}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/DeleteReference"
        super().__init__(handler=_delete_reference, **kwargs)


class GetGitCommit(Tool):
    name: str = "github_get_git_commit"
    description: str | None = "Get a git commit object (tree, parents, author, committer) by SHA."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_git_commit(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            commit_sha: str = Field(..., description="Git commit SHA to retrieve"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/git/commits/{commit_sha}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetGitCommit"
        super().__init__(handler=_get_git_commit, **kwargs)


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


class AddUserToOrganization(Tool):
    name: str = "github_add_user_to_organization"
    description: str | None = "Add or update a user's membership in a GitHub organization."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_user_to_organization(
            org: str = Field(..., description="GitHub organization name"),
            username: str = Field(..., description="GitHub username to add to organization"),
            role: str = Field("member", description="Role for the user: 'member' or 'admin'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_GITHUB_BASE}/orgs/{org}/memberships/{username}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"role": role},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/AddUserToOrganization"
        super().__init__(handler=_add_user_to_organization, **kwargs)


class RemoveUserFromOrganization(Tool):
    name: str = "github_remove_user_from_organization"
    description: str | None = "Remove a user's membership from a GitHub organization."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_user_from_organization(
            org: str = Field(..., description="GitHub organization name"),
            username: str = Field(..., description="GitHub username to remove from organization")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_GITHUB_BASE}/orgs/{org}/members/{username}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"removed": True, "username": username, "org": org}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/RemoveUserFromOrganization"
        super().__init__(handler=_remove_user_from_organization, **kwargs)


class AddUserToTeam(Tool):
    name: str = "github_add_user_to_team"
    description: str | None = "Add or update a user's membership in a GitHub team."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_user_to_team(
            org: str = Field(..., description="GitHub organization name"),
            team_slug: str = Field(..., description="GitHub team slug"),
            username: str = Field(..., description="GitHub username to add to team"),
            role: str = Field("member", description="Role for the user: 'member' or 'maintainer'"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_GITHUB_BASE}/orgs/{org}/teams/{team_slug}/memberships/{username}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"role": role},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/AddUserToTeam"
        super().__init__(handler=_add_user_to_team, **kwargs)


# ---------------------------------------------------------------------------
# Pull Requests
# ---------------------------------------------------------------------------


class ListPullRequests(Tool):
    name: str = "github_list_pull_requests"
    description: str | None = "List pull requests for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_requests(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            state: str = Field("open", description="Pull request state: 'open', 'closed', or 'all'"),
            head: str | None = Field(None, description="Filter by head branch, e.g. 'user:branch-name'"),
            base: str | None = Field(None, description="Filter by base branch, e.g. 'main'"),
            sort: str = Field("created", description="Sort field: 'created', 'updated', 'popularity', or 'long-running'"),
            direction: str = Field("desc", description="Sort direction: 'asc' or 'desc'"),
            per_page: int = Field(30, description="Number of pull requests per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": per_page,
                "page": page,
            }
            if head:
                params["head"] = head
            if base:
                params["base"] = base

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListPullRequests"
        super().__init__(handler=_list_pull_requests, **kwargs)


class GetPullRequest(Tool):
    name: str = "github_get_pull_request"
    description: str | None = "Get a specific pull request by number."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_pull_request(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetPullRequest"
        super().__init__(handler=_get_pull_request, **kwargs)


class GetPullRequestPreview(Tool):
    name: str = "github_get_pull_request_preview"
    description: str | None = "Get the unified diff (patch) of a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_pull_request_preview(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github.diff",
                    },
                )
                response.raise_for_status()
                return {"diff": response.text}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetPullRequestPreview"
        super().__init__(handler=_get_pull_request_preview, **kwargs)


class CreatePullRequest(Tool):
    name: str = "github_create_pull_request"
    description: str | None = "Create a new pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_pull_request(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            title: str = Field(..., description="Pull request title"),
            head: str = Field(..., description="Branch containing your changes, e.g. 'feature-branch' or 'user:feature-branch'"),
            base: str = Field(..., description="Branch you want to merge into, e.g. 'main'"),
            body: str | None = Field(None, description="Pull request description"),
            draft: bool = Field(False, description="Whether the pull request should be created as a draft"),
            maintainer_can_modify: bool = Field(True, description="Whether the maintainer can modify the pull request"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            payload: dict[str, Any] = {
                "title": title,
                "head": head,
                "base": base,
                "draft": draft,
                "maintainer_can_modify": maintainer_can_modify,
            }
            if body:
                payload["body"] = body

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_repo(owner, repo)}/pulls",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CreatePullRequest"
        super().__init__(handler=_create_pull_request, **kwargs)


class MergePullRequest(Tool):
    name: str = "github_merge_pull_request"
    description: str | None = "Merge a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _merge_pull_request(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number"),
            commit_title: str | None = Field(None, description="Commit title"),
            commit_message: str | None = Field(None, description="Commit message"),
            merge_method: str = Field("merge", description="Merge method: \"merge\", \"squash\", or \"rebase\"."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            payload: dict[str, Any] = {"merge_method": merge_method}
            if commit_title:
                payload["commit_title"] = commit_title
            if commit_message:
                payload["commit_message"] = commit_message

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/merge",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/MergePullRequest"
        super().__init__(handler=_merge_pull_request, **kwargs)


class CheckIfPullRequestIsMerged(Tool):
    name: str = "github_check_if_pull_request_is_merged"
    description: str | None = "Check whether a pull request has been merged."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check_if_pull_request_is_merged(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/merge",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return {"merged": response.status_code == 204}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CheckIfPullRequestIsMerged"
        super().__init__(handler=_check_if_pull_request_is_merged, **kwargs)


class ListPullRequestCommits(Tool):
    name: str = "github_list_pull_request_commits"
    description: str | None = "List commits on a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_request_commits(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number"),
            per_page: int = Field(30, description="Number of items per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/commits",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListPullRequestCommits"
        super().__init__(handler=_list_pull_request_commits, **kwargs)


class ListPullRequestFiles(Tool):
    name: str = "github_list_pull_request_files"
    description: str | None = "List files changed in a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_request_files(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number"),
            per_page: int = Field(30, description="Number of items per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/files",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListPullRequestFiles"
        super().__init__(handler=_list_pull_request_files, **kwargs)


class ListPullRequestReviews(Tool):
    name: str = "github_list_pull_request_reviews"
    description: str | None = "List reviews for a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_request_reviews(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number"),
            per_page: int = Field(30, description="Number of items per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/reviews",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListPullRequestReviews"
        super().__init__(handler=_list_pull_request_reviews, **kwargs)


class ListPRReviewComments(Tool):
    name: str = "github_list_pr_review_comments"
    description: str | None = "List all inline review comments on a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pr_review_comments(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            pull_number: int = Field(..., description="Pull request number"),
            sort: str = Field("created", description="Sort order"),
            direction: str = Field("asc", description="Sort direction"),
            per_page: int = Field(30, description="Number of items per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/comments",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"sort": sort, "direction": direction, "per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListPRReviewComments"
        super().__init__(handler=_list_pr_review_comments, **kwargs)


class GetReviewComment(Tool):
    name: str = "github_get_review_comment"
    description: str | None = "Get a single inline review comment on a pull request."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_review_comment(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            comment_id: int = Field(..., description="Review comment ID")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/comments/{comment_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetReviewComment"
        super().__init__(handler=_get_review_comment, **kwargs)


class ListCommentsForReview(Tool):
    name: str = "github_list_comments_for_review"
    description: str | None = "List all comments for a specific pull request review."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_comments_for_review(
            owner: str = Field(..., description="GitHub repository owner"), 
            repo: str = Field(..., description="GitHub repository name"), 
            pull_number: int = Field(..., description="Pull request number"), 
            review_id: int = Field(..., description="Review ID"), 
            per_page: int = Field(30, description="Number of comments per page (max 100)"), 
            page: int = Field(1, description="Page number for pagination")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/reviews/{review_id}/comments",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListCommentsForReview"
        super().__init__(handler=_list_comments_for_review, **kwargs)


class ListPullRequestsForCommit(Tool):
    name: str = "github_list_pull_requests_for_commit"
    description: str | None = "List pull requests associated with a specific commit."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_requests_for_commit(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            commit_sha: str = Field(..., description="Commit SHA"),
            per_page: int = Field(30, description="Number of pull requests per page (max 100)"), 
            page: int = Field(1, description="Page number for pagination")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/commits/{commit_sha}/pulls",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github.groot-preview+json",
                    },
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListPullRequestsForCommit"
        super().__init__(handler=_list_pull_requests_for_commit, **kwargs)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class GetRepositoryDetails(Tool):
    name: str = "github_get_repository_details"
    description: str | None = "Get metadata and details for a GitHub repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_repository_details(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _repo(owner, repo),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetRepositoryDetails"
        super().__init__(handler=_get_repository_details, **kwargs)


class CompareBranches(Tool):
    name: str = "github_compare_branches"
    description: str | None = "Compare two branches, tags, or commits and return their diff."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _compare_branches(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            base: str = Field(..., description="Base branch, tag, or commit SHA"),
            head: str = Field(..., description="Head branch, tag, or commit SHA")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/compare/{base}...{head}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CompareBranches"
        super().__init__(handler=_compare_branches, **kwargs)


class GetFileContent(Tool):
    name: str = "github_get_file_content"
    description: str | None = "Get the content of a file or directory in a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file_content(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            path: str = Field(..., description="File path relative to the repository root."),
            ref: str | None = Field(None, description="Branch, tag, or commit SHA. Defaults to the default branch."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if ref:
                params["ref"] = ref

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/contents/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetFileContent"
        super().__init__(handler=_get_file_content, **kwargs)


class CreateOrUpdateFile(Tool):
    name: str = "github_create_or_update_file"
    description: str | None = "Create or update a file in a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_or_update_file(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            path: str = Field(..., description="File path relative to the repository root."),
            message: str = Field(..., description="Commit message."),
            content: str = Field(..., description="Base64-encoded file content."),
            sha: str | None = Field(None, description="Blob SHA of the file being replaced (required when updating an existing file)."),
            branch: str | None = Field(None, description="Branch to commit to. Defaults to the default branch."),
            author_name: str | None = Field(None, description="Author name for the commit."),
            author_email: str | None = Field(None, description="Author email for the commit."),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            payload: dict[str, Any] = {"message": message, "content": content}
            if sha:
                payload["sha"] = sha
            if branch:
                payload["branch"] = branch
            if author_name and author_email:
                payload["author"] = {"name": author_name, "email": author_email}

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_repo(owner, repo)}/contents/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CreateOrUpdateFile"
        super().__init__(handler=_create_or_update_file, **kwargs)


class GetReadme(Tool):
    name: str = "github_get_readme"
    description: str | None = "Get the README file of a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_readme(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            ref: str | None = Field(None, description="Git reference (branch, tag, or commit SHA). If not provided, returns default branch README"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if ref:
                params["ref"] = ref

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/readme",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetReadme"
        super().__init__(handler=_get_readme, **kwargs)


class ListBranches(Tool):
    name: str = "github_list_branches"
    description: str | None = "List branches in a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_branches(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            protected: bool | None = Field(None, description="Filter for protected branches only (true) or non-protected branches only (false). If not provided, returns all branches"),
            per_page: int = Field(30, description="Number of branches per page (max 100)"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if protected is not None:
                params["protected"] = protected

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/branches",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListBranches"
        super().__init__(handler=_list_branches, **kwargs)


class SearchBranches(Tool):
    name: str = "github_search_branches"
    description: str | None = "Search branches in a repository by name pattern."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_branches(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            query: str = Field(..., description="Substring to match against branch names (case-insensitive)"),
            per_page: int = Field(30, description="Number of branches per page (max 100)"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_GITHUB_BASE}/search/refs",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                    },
                    params={
                        "q": f"{query} repo:{owner}/{repo} type:branch",
                        "per_page": per_page,
                        "page": page,
                    },
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/SearchBranches"
        super().__init__(handler=_search_branches, **kwargs)


class ListCommits(Tool):
    name: str = "github_list_commits"
    description: str | None = "List commits for a repository, optionally filtered by branch, path, or author."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_commits(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            sha: str | None = Field(None, description="SHA or branch to start listing from"),
            path: str | None = Field(None, description="Only commits containing this file path"),
            author: str | None = Field(None, description="Only commits by this author (username or email)"),
            since: str | None = Field(None, description="Only commits after this ISO 8601 timestamp e.g. '2024-01-01T00:00:00Z'"),
            until: str | None = Field(None, description="Only commits before this ISO 8601 timestamp e.g. '2024-01-01T00:00:00Z'"),
            per_page: int = Field(30, description="Number of commits per page (max 100)"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if sha:
                params["sha"] = sha
            if path:
                params["path"] = path
            if author:
                params["author"] = author
            if since:
                params["since"] = since
            if until:
                params["until"] = until

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/commits",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListCommits"
        super().__init__(handler=_list_commits, **kwargs)


class ListCommitStatuses(Tool):
    name: str = "github_list_commit_statuses"
    description: str | None = "List commit statuses (CI checks) for a specific ref."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_commit_statuses(
            owner: str, repo: str, ref: str, per_page: int = 30, page: int = 1
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/commits/{ref}/statuses",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListCommitStatuses"
        super().__init__(handler=_list_commit_statuses, **kwargs)


class ListContributors(Tool):
    name: str = "github_list_contributors"
    description: str | None = "List contributors to a repository sorted by number of commits."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_contributors(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            anon: bool = Field(False, description="Include anonymous contributors"),
            per_page: int = Field(30, description="Number of contributors per page"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/contributors",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"anon": anon, "per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListContributors"
        super().__init__(handler=_list_contributors, **kwargs)


class ListCollaborators(Tool):
    name: str = "github_list_collaborators"
    description: str | None = "List collaborators for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_collaborators(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            affiliation: str = Field("all", description="Collaborator affiliation: 'outside', 'direct', or 'all'"),
            permission: str | None = Field(None, description="Filter by permission level: 'pull', 'triage', 'push', 'maintain', 'admin'"),
            per_page: int = Field(30, description="Number of collaborators per page"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"affiliation": affiliation, "per_page": per_page, "page": page}
            if permission:
                params["permission"] = permission

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/collaborators",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListCollaborators"
        super().__init__(handler=_list_collaborators, **kwargs)


class ListTags(Tool):
    name: str = "github_list_tags"
    description: str | None = "List tags in a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tags(
            owner: str = Field(..., description="GitHub repository owner"), 
            repo: str = Field(..., description="GitHub repository name"), 
            per_page: int = Field(30, description="Number of tags per page (max 100)"), 
            page: int = Field(1, description="Page number for pagination")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/tags",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListTags"
        super().__init__(handler=_list_tags, **kwargs)


class ListForks(Tool):
    name: str = "github_list_forks"
    description: str | None = "List forks of a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_forks(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            sort: str = Field("newest", description="Sort order: 'newest', 'oldest', 'stargazers', or 'watchers'"),
            per_page: int = Field(30, description="Number of forks per page (max 100)"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/forks",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"sort": sort, "per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListForks"
        super().__init__(handler=_list_forks, **kwargs)


class ListDeployments(Tool):
    name: str = "github_list_deployments"
    description: str | None = "List deployments for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployments(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            sha: str | None = Field(None, description="Commit SHA to filter deployments"),
            ref: str | None = Field(None, description="Branch or tag name to filter deployments"),
            task: str | None = Field(None, description="Task name to filter deployments"),
            environment: str | None = Field(None, description="Environment name to filter deployments"),
            per_page: int = Field(30, description="Number of deployments per page (max 100)"),
            page: int = Field(1, description="Page number for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if sha:
                params["sha"] = sha
            if ref:
                params["ref"] = ref
            if task:
                params["task"] = task
            if environment:
                params["environment"] = environment

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/deployments",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListDeployments"
        super().__init__(handler=_list_deployments, **kwargs)


class ListActivities(Tool):
    name: str = "github_list_activities"
    description: str | None = "List activity events for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_activities(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            direction: str = Field("desc", description="Sort order: 'asc' or 'desc'"),
            per_page: int = Field(30, description="Number of activities per page (max 100)"),
            before: str | None = Field(None, description="ISO 8601 timestamp to filter activities before"),
            after: str | None = Field(None, description="ISO 8601 timestamp to filter activities after"),
            ref: str | None = Field(None, description="Git ref to filter activities"),
            actor: str | None = Field(None, description="GitHub username to filter activities"),
            activity_type: str | None = Field(None, description="Activity type to filter (e.g., 'push', 'force_push', 'branch_creation', 'branch_deletion', 'pr_merge', 'merge_queue_merge')"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"direction": direction, "per_page": per_page}
            if before:
                params["before"] = before
            if after:
                params["after"] = after
            if ref:
                params["ref"] = ref
            if actor:
                params["actor"] = actor
            if activity_type:
                params["activity_type"] = activity_type

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/activity",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListActivities"
        super().__init__(handler=_list_activities, **kwargs)


class ListTeams(Tool):
    name: str = "github_list_teams"
    description: str | None = "List teams that have access to a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_teams(
            owner: str = Field(..., description="GitHub repository owner"), 
            repo: str = Field(..., description="GitHub repository name"), 
            per_page: int = Field(30, description="Number of teams per page (max 100)"), 
            page: int = Field(1, description="Page number for pagination")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/teams",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListTeams"
        super().__init__(handler=_list_teams, **kwargs)


# ---------------------------------------------------------------------------
# Releases & Pages
# ---------------------------------------------------------------------------


class ListReleases(Tool):
    name: str = "github_list_releases"
    description: str | None = "List releases for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_releases(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            per_page: int = Field(30, description="Number of releases per page (max 100)"),
            page: int = Field(1, description="Page number for pagination")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/releases",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListReleases"
        super().__init__(handler=_list_releases, **kwargs)


class GetRelease(Tool):
    name: str = "github_get_release"
    description: str | None = "Get a specific release by ID."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_release(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            release_id: int = Field(..., description="Release ID")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/releases/{release_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetRelease"
        super().__init__(handler=_get_release, **kwargs)


class GetLatestRelease(Tool):
    name: str = "github_get_latest_release"
    description: str | None = "Get the latest published release for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_latest_release(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/releases/latest",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetLatestRelease"
        super().__init__(handler=_get_latest_release, **kwargs)


class GetPages(Tool):
    name: str = "github_get_pages"
    description: str | None = "Get information about a repository's GitHub Pages site."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_pages(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pages",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetPages"
        super().__init__(handler=_get_pages, **kwargs)


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


class ListWebhooks(Tool):
    name: str = "github_list_webhooks"
    description: str | None = "List webhooks for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_webhooks(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            per_page: int = Field(30, description="Number of webhooks per page (max 100)"),
            page: int = Field(1, description="Page number for pagination")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/hooks",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"per_page": per_page, "page": page},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/ListWebhooks"
        super().__init__(handler=_list_webhooks, **kwargs)


class GetWebhook(Tool):
    name: str = "github_get_webhook"
    description: str | None = "Get a specific webhook for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_webhook(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            hook_id: int = Field(..., description="Webhook ID")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/hooks/{hook_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetWebhook"
        super().__init__(handler=_get_webhook, **kwargs)


# ---------------------------------------------------------------------------
# Actions — Workflows
# ---------------------------------------------------------------------------


class GetWorkflowRun(Tool):
    name: str = "github_get_workflow_run"
    description: str | None = "Get details for a specific GitHub Actions workflow run."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workflow_run(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            run_id: int = Field(..., description="Workflow run ID")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/actions/runs/{run_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetWorkflowRun"
        super().__init__(handler=_get_workflow_run, **kwargs)


class RunWorkflow(Tool):
    name: str = "github_run_workflow"
    description: str | None = "Trigger a GitHub Actions workflow dispatch event."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _run_workflow(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            workflow_id: str = Field(..., description="Workflow file name (e.g. 'deploy.yml') or numeric workflow ID"),
            ref: str = Field(..., description="Branch or tag to run the workflow on"),
            inputs: dict[str, str] | None = Field(None, description="Key-value pairs matching the workflow's on.workflow_dispatch.inputs"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            payload: dict[str, Any] = {"ref": ref}
            if inputs:
                payload["inputs"] = inputs

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_repo(owner, repo)}/actions/workflows/{workflow_id}/dispatches",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"triggered": True, "workflow_id": workflow_id, "ref": ref}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/RunWorkflow"
        super().__init__(handler=_run_workflow, **kwargs)


class SearchWorkflowRuns(Tool):
    name: str = "github_search_workflow_runs"
    description: str | None = "List and filter GitHub Actions workflow runs for a repository."
    integration: Annotated[str, Integration("github")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_workflow_runs(
            owner: str = Field(..., description="GitHub repository owner"),
            repo: str = Field(..., description="GitHub repository name"),
            workflow_id: str | None = Field(None, description="Filter by specific workflow file name or ID"),
            actor: str | None = Field(None, description="Filter by actor who triggered the run"),
            branch: str | None = Field(None, description="Filter by branch"),
            event: str | None = Field(None, description="Filter by event type: e.g. 'push', 'pull_request', 'workflow_dispatch'"),
            status: str | None = Field(None, description="Filter by run status: 'completed', 'action_required', 'cancelled', 'failure', 'neutral', 'skipped', 'stale', 'success', 'timed_out', 'in_progress', 'queued', 'waiting'"),
            created: str | None = Field(None, description="Filter by creation date range (e.g. '>=2024-01-01' or '2024-01-01..2024-12-31')"),
            per_page: int = Field(30, description="Number of runs per page"),
            page: int = Field(1, description="Page number"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if actor:
                params["actor"] = actor
            if branch:
                params["branch"] = branch
            if event:
                params["event"] = event
            if status:
                params["status"] = status
            if created:
                params["created"] = created

            base_url = (
                f"{_repo(owner, repo)}/actions/workflows/{workflow_id}/runs"
                if workflow_id
                else f"{_repo(owner, repo)}/actions/runs"
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    base_url,
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/SearchWorkflowRuns"
        super().__init__(handler=_search_workflow_runs, **kwargs)
