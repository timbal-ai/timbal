from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_DRIVE_BASE = "https://www.googleapis.com/drive/v3"
_DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError(
        "Google Drive credentials not found. Configure an integration or pass token."
    )


class GoogleDriveCreateFile(Tool):
    name: str = "google_drive_create_file"
    description: str | None = "Create a new file in Google Drive with specified name and content."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_file(
            name: str = Field(..., description="Name of the file to create"),
            content: str = Field(..., description="Content of the file to create"),
            folder_id: str | None = Field(None, description="ID of the folder to create the file in (defaults to root)"),
            mime_type: str = Field("text/plain", description="MIME type of the file"),
        ) -> dict:
            token = await _resolve_token(self)
            import json

            import httpx

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
                    f"{_DRIVE_UPLOAD_BASE}/files",
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

        super().__init__(handler=_create_file, **kwargs)


class GoogleDriveGetDownloadLink(Tool):
    name: str = "google_drive_get_download_link"
    description: str | None = (
        "Get download link for a Google Drive file. Returns view links and download URLs. "
        "Note: These links only work when the file is shared with 'Anyone with the link' or when the user is logged in to the owning Google account. "
        "For programmatic access to private files, use google_drive_get_file instead."
    )
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_download_link(
            file_id: str = Field(..., description="ID of the file to get download link for"),
            drive: str | None = Field(None, description="ID of the shared drive (optional)"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "fields": "id,name,webViewLink,webContentLink",
                "supportsAllDrives": "true",
            }
            
            if drive:
                params["driveId"] = drive
                params["includeItemsFromAllDrives"] = "true"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_DRIVE_BASE}/files/{file_id}",
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
                    "directDownloadUrl": f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                    "note": "Links work only when the file is shared with 'Anyone with the link' or when accessed by the owner. Use google_drive_get_file for programmatic access to private files.",
                }

        super().__init__(handler=_get_download_link, **kwargs)


class GoogleDriveGetFile(Tool):
    name: str = "google_drive_get_file"
    description: str | None = "Download and return content of a Google Drive file."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file(
            file_id: str = Field(..., description="ID of the file to download"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_DRIVE_BASE}/files/{file_id}",
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

        super().__init__(handler=_get_file, **kwargs)


class GoogleDriveCreateFolder(Tool):
    name: str = "google_drive_create_folder"
    description: str | None = "Create a new folder in Google Drive with specified name."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_folder(
            name: str = Field(..., description="Name of the folder to create"),
            parent_folder_id: str | None = Field(None, description="ID of the parent folder (defaults to root)"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            folder_metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder"
            }
            
            if parent_folder_id:
                folder_metadata["parents"] = [parent_folder_id]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DRIVE_BASE}/files",
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

        super().__init__(handler=_create_folder, **kwargs)


class GoogleDriveSearchFolders(Tool):
    name: str = "google_drive_search_folders"
    description: str | None = "Search for folders in Google Drive. Optionally limit search to a specific parent folder."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_folders(
            folder_id: str | None = Field(None, description="ID of the parent folder to search in (optional)"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

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
                        f"{_DRIVE_BASE}/files",
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

        super().__init__(handler=_search_folders, **kwargs)


class GoogleDriveSearchFiles(Tool):
    name: str = "google_drive_search_files"
    description: str | None = "Search for files in Google Drive with optional query. Supports pagination, filtering by folder, name, and trashed status."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_files(
            query: str | None = Field(None, description="Search query string"),
            folder_id: str | None = Field(None, description="ID of the folder to search in (optional)"),
            fields: str | None = Field(None, description="Fields to return in response"),
            filter_text: str | None = Field(None, description="Text to filter file names by"),
            filter_type: str = Field("CONTAINS", description="Filter type: CONTAINS or EXACT"),
            trashed: bool | None = Field(None, description="Include trashed files"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            query_parts = []
            if query:
                if "mimeType" not in query:
                    query_parts.append(f"{query} and mimeType!='application/vnd.google-apps.folder'")
                else:
                    query_parts.append(query)
            else:
                if folder_id:
                    query_parts.append(f'"{folder_id}" in parents')
                else:
                    query_parts.append("'me' in parents")
                query_parts.append("mimeType!='application/vnd.google-apps.folder'")

            if filter_text:
                if filter_type == "CONTAINS":
                    query_parts.append(f"name contains '{filter_text}'")
                else:
                    query_parts.append(f"name='{filter_text}'")

            if trashed is not None:
                query_parts.append(f"trashed={str(trashed).lower()}")

            query_string = " and ".join(query_parts)
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
                        f"{_DRIVE_BASE}/files",
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

        super().__init__(handler=_search_files, **kwargs)


class GoogleDriveUploadFile(Tool):
    name: str = "google_drive_upload_file"
    description: str | None = "Upload a file to Google Drive from URL or local path. Supports file replacement and metadata."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            file_url: str = Field(..., description="URL of the file to upload"),
            folder_id: str | None = Field(None, description="ID of the folder to upload to (defaults to root)"),
            file_name: str | None = Field(None, description="Name to save the file as (optional)"),
        ) -> dict:
            token = await _resolve_token(self)
            import json

            import httpx

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
                    f"{_DRIVE_UPLOAD_BASE}/files",
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

        super().__init__(handler=_upload_file, **kwargs)


class GoogleDriveCreateSharedDrive(Tool):
    name: str = "google_drive_create_shared_drive"
    description: str | None = "Create a new shared drive."
    integration: Annotated[str, Integration("google_drive")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_shared_drive(
            name: str = Field(..., description="Name of the shared drive to create"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            drive_metadata = {
                "name": name
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DRIVE_BASE}/drives",
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

        super().__init__(handler=_create_shared_drive, **kwargs)
