from typing import Annotated, Any

from pydantic import Field

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://graph.microsoft.com/v1.0"


async def _resolve_token(tool: Any) -> str:
    """Resolve OneDrive OAuth token from integration."""
    if not isinstance(getattr(tool, "integration", None), Integration):
        raise ValueError("OneDrive integration not configured.")
    credentials = await tool.integration.resolve()
    return credentials["token"]


class OneDriveSearchFiles(Tool):
    name: str = "onedrive_search_files"
    description: str | None = "Search for files in OneDrive using a query string."
    integration: Annotated[str, Integration("onedrive")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_files(query: str = Field(..., description="Search query string to find files in OneDrive")) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/me/drive/root/search(q='{query}')",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_files, **kwargs)


class OneDriveUploadFile(Tool):
    name: str = "onedrive_upload_file"
    description: str | None = "Upload a file to OneDrive with specified name and content. Can also download from URL and upload."
    integration: Annotated[str, Integration("onedrive")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            name: str = Field(..., description="Name of the file to upload to OneDrive"),
            content: str = Field(..., description="Content of the file to upload"),
            folder_id: str | None = Field(None, description="Optional folder ID where the file should be uploaded"),
            url: str | None = Field(None, description="Optional URL to download content from before uploading"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            if url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=httpx.Timeout(30.0, read=None))
                    response.raise_for_status()
                    content = response.text

            upload_url = f"{_BASE_URL}/me/drive/root:/{name}:/content"
            if folder_id:
                upload_url = f"{_BASE_URL}/me/drive/items/{folder_id}:/{name}:/content"

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    upload_url,
                    headers={"Authorization": f"Bearer {token}"},
                    content=content,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_upload_file, **kwargs)


class OneDriveListFiles(Tool):
    name: str = "onedrive_list_files"
    description: str | None = "List files and folders in a specified OneDrive folder."
    integration: Annotated[str, Integration("onedrive")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_files(
            folder_id: str | None = Field(None, description="Optional folder ID to list files from. If not provided, lists files from root directory")
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            url = f"{_BASE_URL}/me/drive/root/children"
            if folder_id:
                url = f"{_BASE_URL}/me/drive/items/{folder_id}/children"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_files, **kwargs)


class OneDriveGetFile(Tool):
    name: str = "onedrive_get_file"
    description: str | None = "Get file metadata and content by file ID from OneDrive."
    integration: Annotated[str, Integration("onedrive")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file(file_id: str = Field(..., description="OneDrive file ID to retrieve metadata and content for")) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                metadata_response = await client.get(
                    f"{_BASE_URL}/me/drive/items/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                metadata_response.raise_for_status()

                content_response = await client.get(
                    f"{_BASE_URL}/me/drive/items/{file_id}/content",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                content_response.raise_for_status()

                return {
                    "metadata": metadata_response.json(),
                    "content": content_response.text,
                }

        super().__init__(handler=_get_file, **kwargs)


class OneDriveFindFile(Tool):
    name: str = "onedrive_find_file"
    description: str | None = "Find a file in OneDrive by its exact name."
    integration: Annotated[str, Integration("onedrive")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_file(
            name: str = Field(..., description="Exact name of the file to find in OneDrive"),
            folder_id: str | None = Field(None, description="Optional folder ID to search within. If not provided, searches in root directory"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            search_url = f"{_BASE_URL}/me/drive/root/search(q='{name}')"
            if folder_id:
                search_url = f"{_BASE_URL}/me/drive/items/{folder_id}/search(q='{name}')"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    search_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()

                results = response.json()
                exact_matches = [
                    item for item in results.get("value", [])
                    if item.get("name", "").lower() == name.lower()
                ]

                return {"value": exact_matches}

        super().__init__(handler=_find_file, **kwargs)


class OneDriveDownloadFile(Tool):
    name: str = "onedrive_download_file"
    description: str | None = "Download a file from OneDrive by file ID."
    integration: Annotated[str, Integration("onedrive")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_file(file_id: str = Field(..., description="OneDrive file ID to download")) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    f"{_BASE_URL}/me/drive/items/{file_id}/content",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()

                return {
                    "content": response.text,
                    "headers": dict(response.headers),
                }

        super().__init__(handler=_download_file, **kwargs)