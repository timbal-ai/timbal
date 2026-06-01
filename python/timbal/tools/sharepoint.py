from typing import Annotated, Any

from pydantic import Field

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://graph.microsoft.com/v1.0"


async def _resolve_token(tool: Any) -> str:
    """Resolve SharePoint OAuth token from integration."""
    if not isinstance(getattr(tool, "integration", None), Integration):
        raise ValueError("SharePoint integration not configured.")
    credentials = await tool.integration.resolve()
    return credentials["token"]


class SharePointSearchSites(Tool):
    name: str = "sharepoint_search_sites"
    description: str | None = "Search for SharePoint sites by name or keyword."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_sites(
            query: str = Field(..., description="Search query string to find SharePoint sites"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites",
                    params={"search": query},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_sites, **kwargs)


class SharePointGetSite(Tool):
    name: str = "sharepoint_get_site"
    description: str | None = (
        "Get a SharePoint site by its ID, by hostname:/sites/{site-name}, or 'root' for the tenant root site."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_site(
            site_id: str = Field(
                ...,
                description=(
                    "SharePoint site identifier. Can be a site ID, 'root' for the tenant root site, "
                    "or a path like 'contoso.sharepoint.com:/sites/Marketing'."
                ),
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_site, **kwargs)


class SharePointListDrives(Tool):
    name: str = "sharepoint_list_drives"
    description: str | None = "List all document libraries (drives) available in a SharePoint site."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_drives(
            site_id: str = Field(..., description="SharePoint site ID to list drives from"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/drives",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_drives, **kwargs)


class SharePointListFiles(Tool):
    name: str = "sharepoint_list_files"
    description: str | None = (
        "List files and folders in a SharePoint site's default document library. "
        "Optionally restrict to a specific folder."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_files(
            site_id: str = Field(..., description="SharePoint site ID"),
            folder_id: str | None = Field(
                None,
                description="Optional folder item ID. If not provided, lists items in the drive root.",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            url = f"{_BASE_URL}/sites/{site_id}/drive/root/children"
            if folder_id:
                url = f"{_BASE_URL}/sites/{site_id}/drive/items/{folder_id}/children"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_files, **kwargs)


class SharePointSearchFiles(Tool):
    name: str = "sharepoint_search_files"
    description: str | None = "Search for files within a SharePoint site's default document library."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_files(
            site_id: str = Field(..., description="SharePoint site ID"),
            query: str = Field(..., description="Search query string to find files in the site's drive"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/drive/root/search(q='{query}')",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_files, **kwargs)


class SharePointGetFile(Tool):
    name: str = "sharepoint_get_file"
    description: str | None = "Get file metadata and content by file ID from a SharePoint site."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file(
            site_id: str = Field(..., description="SharePoint site ID"),
            file_id: str = Field(..., description="Drive item ID of the file"),
        ) -> dict:
            token = await _resolve_token(self)
            import base64
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                metadata_response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{file_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                metadata_response.raise_for_status()

                content_response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{file_id}/content",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                content_response.raise_for_status()

                content_type = content_response.headers.get("content-type", "")
                raw = content_response.content
                payload: dict[str, Any] = {
                    "metadata": metadata_response.json(),
                    "content_type": content_type,
                    "content_base64": base64.b64encode(raw).decode("ascii"),
                }
                # Best-effort text decode for textual MIME types (txt, csv, json, xml, html, ...).
                # Office/PDF binaries stay in content_base64 only.
                if content_type.startswith("text/") or any(
                    s in content_type for s in ("json", "xml", "javascript", "csv")
                ):
                    try:
                        payload["content"] = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                return payload

        super().__init__(handler=_get_file, **kwargs)


class SharePointDownloadFile(Tool):
    name: str = "sharepoint_download_file"
    description: str | None = (
        "Download a file from a SharePoint site by file ID. Returns base64-encoded content "
        "(safe for Word, Excel, PDF and other binaries)."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_file(
            site_id: str = Field(..., description="SharePoint site ID"),
            file_id: str = Field(..., description="Drive item ID of the file to download"),
        ) -> dict:
            token = await _resolve_token(self)
            import base64
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{file_id}/content",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                raw = response.content
                payload: dict[str, Any] = {
                    "content_type": content_type,
                    "size": len(raw),
                    "content_base64": base64.b64encode(raw).decode("ascii"),
                }
                if content_type.startswith("text/") or any(
                    s in content_type for s in ("json", "xml", "javascript", "csv")
                ):
                    try:
                        payload["content"] = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                return payload

        super().__init__(handler=_download_file, **kwargs)


class SharePointMoveItem(Tool):
    name: str = "sharepoint_move_item"
    description: str | None = (
        "Move a file or folder to another folder within the same SharePoint drive (and optionally rename it)."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _move_item(
            site_id: str = Field(..., description="SharePoint site ID where the item currently lives"),
            item_id: str = Field(..., description="Drive item ID of the file or folder to move"),
            new_parent_folder_id: str = Field(
                ...,
                description="Drive item ID of the destination folder within the same site/drive",
            ),
            new_name: str | None = Field(
                None,
                description="Optional new name for the item after the move",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            payload: dict[str, Any] = {"parentReference": {"id": new_parent_folder_id}}
            if new_name:
                payload["name"] = new_name

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_move_item, **kwargs)


class SharePointCopyItem(Tool):
    name: str = "sharepoint_copy_item"
    description: str | None = (
        "Copy a file or folder to another folder, optionally on a different SharePoint site/drive. "
        "Microsoft Graph runs the copy asynchronously; this tool returns the monitor URL "
        "(and the value of the Location response header) so progress can be polled."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _copy_item(
            site_id: str = Field(..., description="SharePoint site ID where the source item lives"),
            item_id: str = Field(..., description="Drive item ID of the file or folder to copy"),
            target_parent_folder_id: str = Field(..., description="Drive item ID of the destination folder"),
            target_drive_id: str | None = Field(
                None,
                description="Optional drive ID of the destination (use to copy to a different site/drive)",
            ),
            new_name: str | None = Field(None, description="Optional new name for the copied item"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            parent_reference: dict[str, Any] = {"id": target_parent_folder_id}
            if target_drive_id:
                parent_reference["driveId"] = target_drive_id

            payload: dict[str, Any] = {"parentReference": parent_reference}
            if new_name:
                payload["name"] = new_name

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}/copy",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return {
                    "status_code": response.status_code,
                    "monitor_url": response.headers.get("Location"),
                    "headers": dict(response.headers),
                }

        super().__init__(handler=_copy_item, **kwargs)


class SharePointListPermissions(Tool):
    name: str = "sharepoint_list_permissions"
    description: str | None = (
        "List all permissions (direct grants and sharing links) on a SharePoint file or folder."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_permissions(
            site_id: str = Field(..., description="SharePoint site ID"),
            item_id: str = Field(..., description="Drive item ID of the file or folder"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}/permissions",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_permissions, **kwargs)


class SharePointInvite(Tool):
    name: str = "sharepoint_invite"
    description: str | None = (
        "Grant access to a SharePoint file or folder by inviting one or more users via email."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _invite(
            site_id: str = Field(..., description="SharePoint site ID"),
            item_id: str = Field(..., description="Drive item ID of the file or folder to share"),
            emails: list[str] = Field(..., description="Email addresses of the recipients"),
            roles: list[str] = Field(
                default_factory=lambda: ["read"],
                description="Roles to grant. Allowed values: 'read', 'write', 'owner'.",
            ),
            message: str | None = Field(None, description="Optional message included in the invitation email"),
            require_sign_in: bool = Field(True, description="Whether recipients must sign in to access the item"),
            send_invitation: bool = Field(True, description="Whether to send an email invitation"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            payload = {
                "recipients": [{"email": email} for email in emails],
                "roles": roles,
                "requireSignIn": require_sign_in,
                "sendInvitation": send_invitation,
            }
            if message:
                payload["message"] = message

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}/invite",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(15.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_invite, **kwargs)


class SharePointCreateShareLink(Tool):
    name: str = "sharepoint_create_share_link"
    description: str | None = (
        "Create a sharing link (anonymous, organization-wide, or for specific users) for a SharePoint item."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_share_link(
            site_id: str = Field(..., description="SharePoint site ID"),
            item_id: str = Field(..., description="Drive item ID of the file or folder"),
            link_type: str = Field(
                "view",
                description="Link type. Allowed values: 'view', 'edit', 'embed'.",
            ),
            scope: str = Field(
                "organization",
                description="Link scope. Allowed values: 'anonymous', 'organization', 'users'.",
            ),
            password: str | None = Field(None, description="Optional password to protect the link"),
            expiration_datetime: str | None = Field(
                None,
                description="Optional ISO 8601 expiration datetime, e.g. '2026-12-31T23:59:59Z'",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            payload: dict[str, Any] = {"type": link_type, "scope": scope}
            if password:
                payload["password"] = password
            if expiration_datetime:
                payload["expirationDateTime"] = expiration_datetime

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}/createLink",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(15.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_share_link, **kwargs)


class SharePointDeletePermission(Tool):
    name: str = "sharepoint_delete_permission"
    description: str | None = "Revoke a permission (direct grant or sharing link) from a SharePoint item."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_permission(
            site_id: str = Field(..., description="SharePoint site ID"),
            item_id: str = Field(..., description="Drive item ID of the file or folder"),
            permission_id: str = Field(..., description="ID of the permission to revoke"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}/permissions/{permission_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted", "permission_id": permission_id}

        super().__init__(handler=_delete_permission, **kwargs)


class SharePointUploadFile(Tool):
    name: str = "sharepoint_upload_file"
    description: str | None = (
        "Upload a file to a SharePoint site's default document library. "
        "Can also download from a URL and upload."
    )
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            site_id: str = Field(..., description="SharePoint site ID"),
            name: str = Field(..., description="Name of the file to create in SharePoint"),
            content: str = Field(..., description="Content of the file to upload"),
            folder_id: str | None = Field(
                None,
                description="Optional parent folder item ID. If not provided, uploads to the drive root.",
            ),
            url: str | None = Field(
                None,
                description="Optional URL to download content from before uploading. Overrides 'content' when provided.",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            if url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=httpx.Timeout(30.0, read=None))
                    response.raise_for_status()
                    content = response.text

            upload_url = f"{_BASE_URL}/sites/{site_id}/drive/root:/{name}:/content"
            if folder_id:
                upload_url = f"{_BASE_URL}/sites/{site_id}/drive/items/{folder_id}:/{name}:/content"

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


class SharePointDeleteItem(Tool):
    name: str = "sharepoint_delete_item"
    description: str | None = "Delete a file or folder from a SharePoint site by item ID."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item(
            site_id: str = Field(..., description="SharePoint site ID"),
            item_id: str = Field(..., description="Drive item ID of the file or folder to delete"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_BASE_URL}/sites/{site_id}/drive/items/{item_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted", "item_id": item_id}

        super().__init__(handler=_delete_item, **kwargs)


class SharePointCreateFolder(Tool):
    name: str = "sharepoint_create_folder"
    description: str | None = "Create a new folder in a SharePoint site's default document library."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_folder(
            site_id: str = Field(..., description="SharePoint site ID"),
            name: str = Field(..., description="Name of the new folder"),
            parent_folder_id: str | None = Field(
                None,
                description="Optional parent folder item ID. If not provided, creates the folder at the drive root.",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            if parent_folder_id:
                url = f"{_BASE_URL}/sites/{site_id}/drive/items/{parent_folder_id}/children"
            else:
                url = f"{_BASE_URL}/sites/{site_id}/drive/root/children"

            payload = {
                "name": name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "rename",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_folder, **kwargs)


class SharePointListLists(Tool):
    name: str = "sharepoint_list_lists"
    description: str | None = "List all SharePoint lists available in a site."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_lists(
            site_id: str = Field(..., description="SharePoint site ID"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/lists",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_lists, **kwargs)


class SharePointGetListItems(Tool):
    name: str = "sharepoint_get_list_items"
    description: str | None = "Get items from a SharePoint list, with their fields expanded."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_list_items(
            site_id: str = Field(..., description="SharePoint site ID"),
            list_id: str = Field(..., description="SharePoint list ID or display name"),
            top: int | None = Field(
                None,
                description="Optional maximum number of items to return (passed as $top to the Graph API).",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"expand": "fields"}
            if top is not None:
                params["$top"] = top

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/sites/{site_id}/lists/{list_id}/items",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_list_items, **kwargs)


class SharePointCreateListItem(Tool):
    name: str = "sharepoint_create_list_item"
    description: str | None = "Create a new item in a SharePoint list."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_list_item(
            site_id: str = Field(..., description="SharePoint site ID"),
            list_id: str = Field(..., description="SharePoint list ID or display name"),
            fields: dict = Field(
                ...,
                description="Field values for the new list item, e.g. {'Title': 'My item', 'Status': 'Open'}.",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            payload = {"fields": fields}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/sites/{site_id}/lists/{list_id}/items",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_list_item, **kwargs)


class SharePointUpdateListItem(Tool):
    name: str = "sharepoint_update_list_item"
    description: str | None = "Update fields of an existing SharePoint list item."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_list_item(
            site_id: str = Field(..., description="SharePoint site ID"),
            list_id: str = Field(..., description="SharePoint list ID or display name"),
            item_id: str = Field(..., description="ID of the list item to update"),
            fields: dict = Field(
                ...,
                description="Fields to update on the list item, e.g. {'Status': 'Closed'}.",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_BASE_URL}/sites/{site_id}/lists/{list_id}/items/{item_id}/fields",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=fields,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_list_item, **kwargs)


class SharePointDeleteListItem(Tool):
    name: str = "sharepoint_delete_list_item"
    description: str | None = "Delete an item from a SharePoint list."
    integration: Annotated[str, Integration("sharepoint")] | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration}, required={"integration"}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_list_item(
            site_id: str = Field(..., description="SharePoint site ID"),
            list_id: str = Field(..., description="SharePoint list ID or display name"),
            item_id: str = Field(..., description="ID of the list item to delete"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_BASE_URL}/sites/{site_id}/lists/{list_id}/items/{item_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return {"status": "deleted", "item_id": item_id}

        super().__init__(handler=_delete_list_item, **kwargs)
