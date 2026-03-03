import json
from typing import Annotated, Any

import httpx
from pydantic import model_validator

from ..core.tool import Tool
from ..platform.integrations import Integration


class GoogleDriveCreateFile(Tool):
    name: str = "google_drive_create_file"
    description: str | None = "Create a new file in Google Drive with specified name and content."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/upload/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveCreateFile":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _create_file(
            name: str,
            content: str,
            folder_id: str | None = None,
            mime_type: str = "text/plain",
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            metadata = {
                "name": name,
                "parents": [folder_id] if folder_id else ["root"]
            }
            
            metadata_json = json.dumps(metadata)
            
            file_content = content.encode()
            
            files = {
                "metadata": (None, metadata_json, "application/json"),
                "file": (name, file_content, mime_type)
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/files",
                    headers={
                        "Authorization": f"Bearer {token}",
                    },
                    params={
                        "uploadType": "multipart",
                        "fields": "id,name,parents,webViewLink"
                    },
                    files=files,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "fileId": data["id"],
                    "fileName": data["name"],
                    "fileUrl": data.get("webViewLink"),
                    "folderId": folder_id or "root"
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/CreateFile"

        super().__init__(handler=_create_file, metadata=metadata, **kwargs)


class GoogleDriveGetDownloadLink(Tool):
    name: str = "google_drive_get_download_link"
    description: str | None = "Get download link for a Google Drive file. Optionally specify a shared drive, or leave empty to use your drive."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveGetDownloadLink":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _get_download_link(
            file_id: str,
            drive: str | None = None,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            params: dict[str, Any] = {
                "fields": "id,name,webViewLink,webContentLink",
                "supportsAllDrives": "true",
            }
            
            if drive:
                params["driveId"] = drive
                params["includeItemsFromAllDrives"] = "true"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/files/{file_id}",
                    headers={
                        "Authorization": f"Bearer {token}",
                    },
                    params=params,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "fileId": data["id"],
                    "fileName": data["name"],
                    "viewLink": data.get("webViewLink"),
                    "downloadLink": data.get("webContentLink"),
                    "directDownloadUrl": f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/GetDownloadLink"

        super().__init__(handler=_get_download_link, metadata=metadata, **kwargs)


class GoogleDriveGetFile(Tool):
    name: str = "google_drive_get_file"
    description: str | None = "Download and return content of a Google Drive file."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveGetFile":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _get_file(
            file_id: str,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/files/{file_id}",
                    headers={
                        "Authorization": f"Bearer {token}",
                    },
                    params={
                        "alt": "media"
                    },
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                content = response.content
                
                return {
                    "fileId": file_id,
                    "content": content.decode('utf-8', errors='ignore'),
                    "contentType": response.headers.get('content-type'),
                    "size": len(content)
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/GetFile"

        super().__init__(handler=_get_file, metadata=metadata, **kwargs)


class GoogleDriveCreateFolder(Tool):
    name: str = "google_drive_create_folder"
    description: str | None = "Create a new folder in Google Drive with specified name."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveCreateFolder":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _create_folder(
            name: str,
            parent_folder_id: str | None = None,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            folder_metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder"
            }
            
            if parent_folder_id:
                folder_metadata["parents"] = [parent_folder_id]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/files",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json=folder_metadata,
                    params={
                        "fields": "id,name,parents"
                    },
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "folderId": data["id"],
                    "folderName": data["name"],
                    "folderUrl": f"https://drive.google.com/drive/folders/{data['id']}"
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/CreateFolder"

        super().__init__(handler=_create_folder, metadata=metadata, **kwargs)


class GoogleDriveSearchFolders(Tool):
    name: str = "google_drive_search_folders"
    description: str | None = "Search for folders in Google Drive. Optionally limit search to a specific parent folder."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveSearchFolders":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _search_folders(
            folder_id: str | None = None,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            # Build query - search for folders only
            query = "mimeType='application/vnd.google-apps.folder'"
            if folder_id:
                query = f'"{folder_id}" in parents and {query}'
            else:
                query = f"'me' in parents and {query}"

            async with httpx.AsyncClient() as client:
                all_folders = []
                page_token = None
                
                while True:
                    params: dict[str, Any] = {
                        "q": query,
                        "fields": "nextPageToken,files(id,name,parents,mimeType)",
                        "pageSize": 1000,
                    }
                    
                    if page_token:
                        params["pageToken"] = page_token
                    
                    response = await client.get(
                        f"{self.base_url}/files",
                        headers={
                            "Authorization": f"Bearer {token}",
                        },
                        params=params,
                        timeout=httpx.Timeout(30.0, read=None),
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for item in data.get("files", []):
                        all_folders.append({
                            "id": item["id"],
                            "name": item["name"],
                            "parent_id": item.get("parents", [None])[0] if item.get("parents") else None
                        })
                    
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
                
                return {
                    "folders": all_folders,
                    "total": len(all_folders)
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/SearchFolders"

        super().__init__(handler=_search_folders, metadata=metadata, **kwargs)


class GoogleDriveSearchFiles(Tool):
    name: str = "google_drive_search_files"
    description: str | None = "Search for files in Google Drive with optional query. Supports pagination, filtering by folder, name, and trashed status."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveSearchFiles":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _search_files(
            query: str | None = None,
            folder_id: str | None = None,
            fields: str | None = None,
            filter_text: str | None = None,
            filter_type: str = "CONTAINS",
            trashed: bool | None = None,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            # Build query string
            query_parts = []
            
            # If custom query provided, use it (but exclude folders)
            if query:
                if "mimeType" not in query:
                    query_parts.append(f"{query} and mimeType!='application/vnd.google-apps.folder'")
                else:
                    query_parts.append(query)
            else:
                # Build query from parameters
                if folder_id:
                    query_parts.append(f'"{folder_id}" in parents')
                else:
                    query_parts.append("'me' in parents")
                
                # Exclude folders
                query_parts.append("mimeType!='application/vnd.google-apps.folder'")
            
            # Add filter text
            if filter_text:
                if filter_type == "CONTAINS":
                    query_parts.append(f"name contains '{filter_text}'")
                else:  # EXACT MATCH
                    query_parts.append(f"name='{filter_text}'")
            
            # Add trashed filter
            if trashed is not None:
                query_parts.append(f"trashed={str(trashed).lower()}")
            
            query_string = " and ".join(query_parts)
            
            # Set fields
            fields_param = fields or "files(id,name,parents,mimeType,webViewLink,modifiedTime,createdTime,size)"
            
            async with httpx.AsyncClient() as client:
                all_files = []
                page_token = None
                
                while True:
                    params: dict[str, Any] = {
                        "q": query_string,
                        "fields": f"nextPageToken,{fields_param}",
                        "pageSize": 1000,
                    }
                    
                    if page_token:
                        params["pageToken"] = page_token
                    
                    response = await client.get(
                        f"{self.base_url}/files",
                        headers={
                            "Authorization": f"Bearer {token}",
                        },
                        params=params,
                        timeout=httpx.Timeout(30.0, read=None),
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for item in data.get("files", []):
                        all_files.append({
                            "id": item["id"],
                            "name": item["name"],
                            "mimeType": item.get("mimeType"),
                            "parent_id": item.get("parents", [None])[0] if item.get("parents") else None,
                            "viewUrl": item.get("webViewLink"),
                            "modifiedTime": item.get("modifiedTime"),
                            "createdTime": item.get("createdTime"),
                            "size": item.get("size"),
                        })
                    
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
                
                return {
                    "files": all_files,
                    "total": len(all_files)
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/SearchFiles"

        super().__init__(handler=_search_files, metadata=metadata, **kwargs)


class GoogleDriveUploadFile(Tool):
    name: str = "google_drive_upload_file"
    description: str | None = "Upload a file to Google Drive from URL or local path. Supports file replacement and metadata."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/upload/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveUploadFile":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
            file_url: str,
            folder_id: str | None = None,
            file_name: str | None = None,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(file_url)
                response.raise_for_status()
                file_content = response.content
                
                if not file_name:
                    file_name = file_url.split("/")[-1] if "/" in file_url else "uploaded_file"

                metadata = {
                    "name": file_name,
                    "parents": [folder_id] if folder_id else ["root"]
                }

                metadata_json = json.dumps(metadata)
                
                files = {
                    "metadata": (None, metadata_json, "application/json"),
                    "file": (file_name, file_content, "application/octet-stream")
                }

                upload_response = await client.post(
                    f"{self.base_url}/files",
                    headers={
                        "Authorization": f"Bearer {token}",
                    },
                    params={
                        "uploadType": "multipart",
                        "fields": "id,name,parents,webViewLink"
                    },
                    files=files,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                upload_response.raise_for_status()
                data = upload_response.json()
                
                return {
                    "fileId": data["id"],
                    "fileName": data["name"],
                    "fileUrl": data.get("webViewLink"),
                    "folderId": folder_id or "root"
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/UploadFile"

        super().__init__(handler=_upload_file, metadata=metadata, **kwargs)


class GoogleDriveCreateSharedDrive(Tool):
    name: str = "google_drive_create_shared_drive"
    description: str | None = "Create a new shared drive."
    integration: Annotated[str, Integration("google_drive")] | None = None
    base_url: str = "https://www.googleapis.com/drive/v3"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDriveCreateSharedDrive":
        if self.integration is None:
            raise ValueError(
                "Google Drive integration not found. Please configure google_drive integration."
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
        async def _create_shared_drive(
            name: str,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            drive_metadata = {
                "name": name
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/drives",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json=drive_metadata,
                    params={
                        "fields": "id,name"
                    },
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "driveId": data["id"],
                    "driveName": data["name"],
                    "summary": f"Successfully created a new shared drive, \"{data['name']}\""
                }

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDrive/CreateSharedDrive"

        super().__init__(handler=_create_shared_drive, metadata=metadata, **kwargs)
