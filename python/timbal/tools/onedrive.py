from typing import Annotated, Any

import httpx
from pydantic import model_validator

from ..core.tool import Tool
from ..platform.integrations import Integration


class OneDriveSearchFiles(Tool):
    name: str = "onedrive_search_files"
    description: str | None = "Search for files in OneDrive using a query string."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveSearchFiles":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_files(query: str) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/me/drive/root/search(q='{query}')",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/SearchFiles"

        super().__init__(handler=_search_files, metadata=metadata, **kwargs)


class OneDriveUploadFile(Tool):
    name: str = "onedrive_upload_file"
    description: str | None = "Upload a file to OneDrive with specified name and content. Can also download from URL and upload."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveUploadFile":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            name: str,
            content: str,
            folder_id: str | None = None,
            url: str | None = None,
        ) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            # If URL is provided, download the content first
            if url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=httpx.Timeout(30.0, read=None))
                    response.raise_for_status()
                    content = response.text

            upload_url = f"{self.base_url}/me/drive/root:/{name}:/content"
            if folder_id:
                upload_url = f"{self.base_url}/me/drive/items/{folder_id}:/{name}:/content"

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    upload_url,
                    headers={"Authorization": f"Bearer {token}"},
                    content=content,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/UploadFile"

        super().__init__(handler=_upload_file, metadata=metadata, **kwargs)


class OneDriveListFiles(Tool):
    name: str = "onedrive_list_files"
    description: str | None = "List files and folders in a specified OneDrive folder."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveListFiles":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_files(folder_id: str | None = None) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            url = f"{self.base_url}/me/drive/root/children"
            if folder_id:
                url = f"{self.base_url}/me/drive/items/{folder_id}/children"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/ListFiles"

        super().__init__(handler=_list_files, metadata=metadata, **kwargs)


class OneDriveGetFile(Tool):
    name: str = "onedrive_get_file"
    description: str | None = "Get file metadata and content by file ID from OneDrive."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveGetFile":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file(file_id: str) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            async with httpx.AsyncClient(follow_redirects=True) as client:
                metadata_response = await client.get(
                    f"{self.base_url}/me/drive/items/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                metadata_response.raise_for_status()
                
                content_response = await client.get(
                    f"{self.base_url}/me/drive/items/{file_id}/content",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                content_response.raise_for_status()
                
                return {
                    "metadata": metadata_response.json(),
                    "content": content_response.text,
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/GetFile"

        super().__init__(handler=_get_file, metadata=metadata, **kwargs)


class OneDriveGetTable(Tool):
    name: str = "onedrive_get_table"
    description: str | None = "Get table data from an Excel file in OneDrive."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveGetTable":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_table(file_id: str, table_name: str | None = None) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            if table_name:
                url = f"{self.base_url}/me/drive/items/{file_id}/workbook/tables/{table_name}"
            else:
                url = f"{self.base_url}/me/drive/items/{file_id}/workbook/tables"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/GetTable"

        super().__init__(handler=_get_table, metadata=metadata, **kwargs)


class OneDriveFindFile(Tool):
    name: str = "onedrive_find_file"
    description: str | None = "Find a file in OneDrive by its exact name."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveFindFile":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_file(name: str, folder_id: str | None = None) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            search_url = f"{self.base_url}/me/drive/root/search(q='{name}')"
            if folder_id:
                search_url = f"{self.base_url}/me/drive/items/{folder_id}/search(q='{name}')"

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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/FindFile"

        super().__init__(handler=_find_file, metadata=metadata, **kwargs)


class OneDriveDownloadFile(Tool):
    name: str = "onedrive_download_file"
    description: str | None = "Download a file from OneDrive by file ID."
    integration: Annotated[str, Integration("onedrive")] | None = None
    base_url: str = "https://graph.microsoft.com/v1.0"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "OneDriveDownloadFile":
        if self.integration is None:
            raise ValueError(
                "OneDrive integration not found. Please configure onedrive integration."
            )
        return self

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_file(file_id: str) -> dict:
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            token = credentials["token"]

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    f"{self.base_url}/me/drive/items/{file_id}/content",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                
                return {
                    "content": response.text,
                    "headers": dict(response.headers),
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "OneDrive/DownloadFile"

        super().__init__(handler=_download_file, metadata=metadata, **kwargs)