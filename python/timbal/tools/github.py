from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_GITHUB_API_BASE = "https://api.github.com"


def _repo(owner: str, repo: str) -> str:
    return f"{_GITHUB_API_BASE}/repos/{owner}/{repo}"


# ---------------------------------------------------------------------------
# Git Data — References & Commits
# ---------------------------------------------------------------------------


class GetReference(Tool):
    name: str = "github_get_reference"
    description: str | None = "Get a single git reference (branch, tag, or HEAD) from a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_reference(owner: str, repo: str, ref: str) -> Any:
            """
            ref: fully-qualified reference, e.g. "heads/main" or "tags/v1.0".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/git/ref/{ref}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetReference"
        super().__init__(handler=_get_reference, metadata=metadata, **kwargs)


class CreateReference(Tool):
    name: str = "github_create_reference"
    description: str | None = "Create a git reference (branch or tag) in a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_reference(owner: str, repo: str, ref: str, sha: str) -> Any:
            """
            ref: fully-qualified reference name, e.g. "refs/heads/my-branch".
            sha: the SHA1 value to set the reference to.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_create_reference, metadata=metadata, **kwargs)


class UpdateReference(Tool):
    name: str = "github_update_reference"
    description: str | None = "Update a git reference to point to a new commit SHA."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_reference(
            owner: str,
            repo: str,
            ref: str,
            sha: str,
            force: bool = False,
        ) -> Any:
            """
            ref: reference to update, e.g. "heads/main".
            force: if True, force-updates even if it's not a fast-forward.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_update_reference, metadata=metadata, **kwargs)


class DeleteReference(Tool):
    name: str = "github_delete_reference"
    description: str | None = "Delete a git reference (branch or tag) from a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_reference(owner: str, repo: str, ref: str) -> Any:
            """
            ref: reference to delete, e.g. "heads/my-branch".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_repo(owner, repo)}/git/refs/{ref}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "ref": ref}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/DeleteReference"
        super().__init__(handler=_delete_reference, metadata=metadata, **kwargs)


class GetGitCommit(Tool):
    name: str = "github_get_git_commit"
    description: str | None = "Get a git commit object (tree, parents, author, committer) by SHA."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_git_commit(owner: str, repo: str, commit_sha: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/git/commits/{commit_sha}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetGitCommit"
        super().__init__(handler=_get_git_commit, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


class AddUserToOrganization(Tool):
    name: str = "github_add_user_to_organization"
    description: str | None = "Add or update a user's membership in a GitHub organization."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_user_to_organization(
            org: str,
            username: str,
            role: str = "member",
        ) -> Any:
            """
            role: "member" or "admin".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_GITHUB_API_BASE}/orgs/{org}/memberships/{username}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"role": role},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/AddUserToOrganization"
        super().__init__(handler=_add_user_to_organization, metadata=metadata, **kwargs)


class RemoveUserFromOrganization(Tool):
    name: str = "github_remove_user_from_organization"
    description: str | None = "Remove a user's membership from a GitHub organization."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_user_from_organization(org: str, username: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_GITHUB_API_BASE}/orgs/{org}/members/{username}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"removed": True, "username": username, "org": org}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/RemoveUserFromOrganization"
        super().__init__(handler=_remove_user_from_organization, metadata=metadata, **kwargs)


class AddUserToTeam(Tool):
    name: str = "github_add_user_to_team"
    description: str | None = "Add or update a user's membership in a GitHub team."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_user_to_team(
            org: str,
            team_slug: str,
            username: str,
            role: str = "member",
        ) -> Any:
            """
            role: "member" or "maintainer".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_GITHUB_API_BASE}/orgs/{org}/teams/{team_slug}/memberships/{username}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"role": role},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/AddUserToTeam"
        super().__init__(handler=_add_user_to_team, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Pull Requests
# ---------------------------------------------------------------------------


class ListPullRequests(Tool):
    name: str = "github_list_pull_requests"
    description: str | None = "List pull requests for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_requests(
            owner: str,
            repo: str,
            state: str = "open",
            head: str | None = None,
            base: str | None = None,
            sort: str = "created",
            direction: str = "desc",
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            """
            state: "open", "closed", or "all".
            head: filter by head branch, e.g. "user:branch-name".
            base: filter by base branch, e.g. "main".
            sort: "created", "updated", "popularity", or "long-running".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_pull_requests, metadata=metadata, **kwargs)


class GetPullRequest(Tool):
    name: str = "github_get_pull_request"
    description: str | None = "Get a specific pull request by number."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_pull_request(owner: str, repo: str, pull_number: int) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetPullRequest"
        super().__init__(handler=_get_pull_request, metadata=metadata, **kwargs)


class GetPullRequestPreview(Tool):
    name: str = "github_get_pull_request_preview"
    description: str | None = "Get the unified diff (patch) of a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_pull_request_preview(owner: str, repo: str, pull_number: int) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_get_pull_request_preview, metadata=metadata, **kwargs)


class CreatePullRequest(Tool):
    name: str = "github_create_pull_request"
    description: str | None = "Create a new pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_pull_request(
            owner: str,
            repo: str,
            title: str,
            head: str,
            base: str,
            body: str | None = None,
            draft: bool = False,
            maintainer_can_modify: bool = True,
        ) -> Any:
            """
            head: branch containing your changes, e.g. "feature-branch" or "user:feature-branch".
            base: branch you want to merge into, e.g. "main".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_create_pull_request, metadata=metadata, **kwargs)


class MergePullRequest(Tool):
    name: str = "github_merge_pull_request"
    description: str | None = "Merge a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _merge_pull_request(
            owner: str,
            repo: str,
            pull_number: int,
            commit_title: str | None = None,
            commit_message: str | None = None,
            merge_method: str = "merge",
        ) -> Any:
            """
            merge_method: "merge", "squash", or "rebase".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_merge_pull_request, metadata=metadata, **kwargs)


class CheckIfPullRequestIsMerged(Tool):
    name: str = "github_check_if_pull_request_is_merged"
    description: str | None = "Check whether a pull request has been merged."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check_if_pull_request_is_merged(
            owner: str, repo: str, pull_number: int
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/{pull_number}/merge",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return {"merged": response.status_code == 204}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CheckIfPullRequestIsMerged"
        super().__init__(handler=_check_if_pull_request_is_merged, metadata=metadata, **kwargs)


class ListPullRequestCommits(Tool):
    name: str = "github_list_pull_request_commits"
    description: str | None = "List commits on a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_request_commits(
            owner: str, repo: str, pull_number: int, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_pull_request_commits, metadata=metadata, **kwargs)


class ListPullRequestFiles(Tool):
    name: str = "github_list_pull_request_files"
    description: str | None = "List files changed in a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_request_files(
            owner: str, repo: str, pull_number: int, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_pull_request_files, metadata=metadata, **kwargs)


class ListPullRequestReviews(Tool):
    name: str = "github_list_pull_request_reviews"
    description: str | None = "List reviews for a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_request_reviews(
            owner: str, repo: str, pull_number: int, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_pull_request_reviews, metadata=metadata, **kwargs)


class ListPRReviewComments(Tool):
    name: str = "github_list_pr_review_comments"
    description: str | None = "List all inline review comments on a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pr_review_comments(
            owner: str,
            repo: str,
            pull_number: int,
            sort: str = "created",
            direction: str = "asc",
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_pr_review_comments, metadata=metadata, **kwargs)


class GetReviewComment(Tool):
    name: str = "github_get_review_comment"
    description: str | None = "Get a single inline review comment on a pull request."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_review_comment(owner: str, repo: str, comment_id: int) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pulls/comments/{comment_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetReviewComment"
        super().__init__(handler=_get_review_comment, metadata=metadata, **kwargs)


class ListCommentsForReview(Tool):
    name: str = "github_list_comments_for_review"
    description: str | None = "List all comments for a specific pull request review."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_comments_for_review(
            owner: str, repo: str, pull_number: int, review_id: int, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_comments_for_review, metadata=metadata, **kwargs)


class ListPullRequestsForCommit(Tool):
    name: str = "github_list_pull_requests_for_commit"
    description: str | None = "List pull requests associated with a specific commit."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_pull_requests_for_commit(
            owner: str, repo: str, commit_sha: str, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_pull_requests_for_commit, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class GetRepositoryDetails(Tool):
    name: str = "github_get_repository_details"
    description: str | None = "Get metadata and details for a GitHub repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_repository_details(owner: str, repo: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _repo(owner, repo),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetRepositoryDetails"
        super().__init__(handler=_get_repository_details, metadata=metadata, **kwargs)


class CompareBranches(Tool):
    name: str = "github_compare_branches"
    description: str | None = "Compare two branches, tags, or commits and return their diff."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _compare_branches(owner: str, repo: str, base: str, head: str) -> Any:
            """
            base: base branch, tag, or commit SHA.
            head: head branch, tag, or commit SHA.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/compare/{base}...{head}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/CompareBranches"
        super().__init__(handler=_compare_branches, metadata=metadata, **kwargs)


class GetFileContent(Tool):
    name: str = "github_get_file_content"
    description: str | None = "Get the content of a file or directory in a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file_content(
            owner: str,
            repo: str,
            path: str,
            ref: str | None = None,
        ) -> Any:
            """
            path: file path relative to the repository root.
            ref: branch, tag, or commit SHA. Defaults to the default branch.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_get_file_content, metadata=metadata, **kwargs)


class CreateOrUpdateFile(Tool):
    name: str = "github_create_or_update_file"
    description: str | None = "Create or update a file in a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_or_update_file(
            owner: str,
            repo: str,
            path: str,
            message: str,
            content: str,
            sha: str | None = None,
            branch: str | None = None,
            author_name: str | None = None,
            author_email: str | None = None,
        ) -> Any:
            """
            content: base64-encoded file content.
            sha: blob SHA of the file being replaced (required when updating an existing file).
            branch: branch to commit to. Defaults to the default branch.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_create_or_update_file, metadata=metadata, **kwargs)


class GetReadme(Tool):
    name: str = "github_get_readme"
    description: str | None = "Get the README file of a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_readme(owner: str, repo: str, ref: str | None = None) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_get_readme, metadata=metadata, **kwargs)


class ListBranches(Tool):
    name: str = "github_list_branches"
    description: str | None = "List branches in a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_branches(
            owner: str,
            repo: str,
            protected: bool | None = None,
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_branches, metadata=metadata, **kwargs)


class SearchBranches(Tool):
    name: str = "github_search_branches"
    description: str | None = "Search branches in a repository by name pattern."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_branches(
            owner: str,
            repo: str,
            query: str,
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            """
            query: substring to match against branch names (case-insensitive).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_GITHUB_API_BASE}/search/refs",
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
        super().__init__(handler=_search_branches, metadata=metadata, **kwargs)


class ListCommits(Tool):
    name: str = "github_list_commits"
    description: str | None = "List commits for a repository, optionally filtered by branch, path, or author."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_commits(
            owner: str,
            repo: str,
            sha: str | None = None,
            path: str | None = None,
            author: str | None = None,
            since: str | None = None,
            until: str | None = None,
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            """
            sha: SHA or branch to start listing from.
            since / until: ISO 8601 timestamps, e.g. "2024-01-01T00:00:00Z".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_commits, metadata=metadata, **kwargs)


class ListCommitStatuses(Tool):
    name: str = "github_list_commit_statuses"
    description: str | None = "List commit statuses (CI checks) for a specific ref."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_commit_statuses(
            owner: str, repo: str, ref: str, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_commit_statuses, metadata=metadata, **kwargs)


class ListContributors(Tool):
    name: str = "github_list_contributors"
    description: str | None = "List contributors to a repository sorted by number of commits."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_contributors(
            owner: str, repo: str, anon: bool = False, per_page: int = 30, page: int = 1
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_contributors, metadata=metadata, **kwargs)


class ListCollaborators(Tool):
    name: str = "github_list_collaborators"
    description: str | None = "List collaborators for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_collaborators(
            owner: str,
            repo: str,
            affiliation: str = "all",
            permission: str | None = None,
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            """
            affiliation: "outside", "direct", or "all".
            permission: filter by permission level: "pull", "triage", "push", "maintain", "admin".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_collaborators, metadata=metadata, **kwargs)


class ListTags(Tool):
    name: str = "github_list_tags"
    description: str | None = "List tags in a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tags(owner: str, repo: str, per_page: int = 30, page: int = 1) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_tags, metadata=metadata, **kwargs)


class ListForks(Tool):
    name: str = "github_list_forks"
    description: str | None = "List forks of a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_forks(
            owner: str,
            repo: str,
            sort: str = "newest",
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            """
            sort: "newest", "oldest", "stargazers", or "watchers".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_forks, metadata=metadata, **kwargs)


class ListDeployments(Tool):
    name: str = "github_list_deployments"
    description: str | None = "List deployments for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployments(
            owner: str,
            repo: str,
            sha: str | None = None,
            ref: str | None = None,
            task: str | None = None,
            environment: str | None = None,
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_deployments, metadata=metadata, **kwargs)


class ListActivities(Tool):
    name: str = "github_list_activities"
    description: str | None = "List activity events for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_activities(
            owner: str,
            repo: str,
            direction: str = "desc",
            per_page: int = 30,
            before: str | None = None,
            after: str | None = None,
            ref: str | None = None,
            actor: str | None = None,
            activity_type: str | None = None,
        ) -> Any:
            """
            activity_type: filter by type, e.g. "push", "force_push", "branch_creation",
                           "branch_deletion", "pr_merge", "merge_queue_merge".
            ref: filter by Git ref.
            actor: filter by GitHub username.
            before / after: ISO 8601 timestamps for date range filtering.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_activities, metadata=metadata, **kwargs)


class ListTeams(Tool):
    name: str = "github_list_teams"
    description: str | None = "List teams that have access to a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_teams(owner: str, repo: str, per_page: int = 30, page: int = 1) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_teams, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Releases & Pages
# ---------------------------------------------------------------------------


class ListReleases(Tool):
    name: str = "github_list_releases"
    description: str | None = "List releases for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_releases(owner: str, repo: str, per_page: int = 30, page: int = 1) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_releases, metadata=metadata, **kwargs)


class GetRelease(Tool):
    name: str = "github_get_release"
    description: str | None = "Get a specific release by ID."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_release(owner: str, repo: str, release_id: int) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/releases/{release_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetRelease"
        super().__init__(handler=_get_release, metadata=metadata, **kwargs)


class GetLatestRelease(Tool):
    name: str = "github_get_latest_release"
    description: str | None = "Get the latest published release for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_latest_release(owner: str, repo: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/releases/latest",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetLatestRelease"
        super().__init__(handler=_get_latest_release, metadata=metadata, **kwargs)


class GetPages(Tool):
    name: str = "github_get_pages"
    description: str | None = "Get information about a repository's GitHub Pages site."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_pages(owner: str, repo: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/pages",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetPages"
        super().__init__(handler=_get_pages, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


class ListWebhooks(Tool):
    name: str = "github_list_webhooks"
    description: str | None = "List webhooks for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_webhooks(owner: str, repo: str, per_page: int = 30, page: int = 1) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_list_webhooks, metadata=metadata, **kwargs)


class GetWebhook(Tool):
    name: str = "github_get_webhook"
    description: str | None = "Get a specific webhook for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_webhook(owner: str, repo: str, hook_id: int) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/hooks/{hook_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetWebhook"
        super().__init__(handler=_get_webhook, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Actions — Workflows
# ---------------------------------------------------------------------------


class GetWorkflowRun(Tool):
    name: str = "github_get_workflow_run"
    description: str | None = "Get details for a specific GitHub Actions workflow run."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workflow_run(owner: str, repo: str, run_id: int) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_repo(owner, repo)}/actions/runs/{run_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GitHub/GetWorkflowRun"
        super().__init__(handler=_get_workflow_run, metadata=metadata, **kwargs)


class RunWorkflow(Tool):
    name: str = "github_run_workflow"
    description: str | None = "Trigger a GitHub Actions workflow dispatch event."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _run_workflow(
            owner: str,
            repo: str,
            workflow_id: str,
            ref: str,
            inputs: dict[str, str] | None = None,
        ) -> Any:
            """
            workflow_id: workflow file name (e.g. "deploy.yml") or numeric workflow ID.
            ref: branch or tag to run the workflow on.
            inputs: key-value pairs matching the workflow's on.workflow_dispatch.inputs.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_run_workflow, metadata=metadata, **kwargs)


class SearchWorkflowRuns(Tool):
    name: str = "github_search_workflow_runs"
    description: str | None = "List and filter GitHub Actions workflow runs for a repository."
    integration: Annotated[str, Integration("github")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_workflow_runs(
            owner: str,
            repo: str,
            workflow_id: str | None = None,
            actor: str | None = None,
            branch: str | None = None,
            event: str | None = None,
            status: str | None = None,
            created: str | None = None,
            per_page: int = 30,
            page: int = 1,
        ) -> Any:
            """
            workflow_id: filter by specific workflow file name or ID.
            status: "completed", "action_required", "cancelled", "failure", "neutral",
                    "skipped", "stale", "success", "timed_out", "in_progress", "queued", "waiting".
            event: filter by trigger event, e.g. "push", "pull_request", "workflow_dispatch".
            created: date range filter, e.g. ">=2024-01-01" or "2024-01-01..2024-12-31".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
        super().__init__(handler=_search_workflow_runs, metadata=metadata, **kwargs)
