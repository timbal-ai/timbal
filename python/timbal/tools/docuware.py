import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_PATH = "/DocuWare/Platform"


def _dw_url(server_url: str, path: str) -> str:
    return f"{server_url.rstrip('/')}{_API_PATH}/{path.lstrip('/')}"


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Return (server_url, token).

    Resolution order for each value:
    1. Integration credentials
    2. Explicit field on the tool
    3. Environment variable

    If no token is available but username/password/organization are, a fresh
    logon is performed against the DocuWare REST API to obtain one.
    """
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    server_url = (
        creds.get("server_url")
        or getattr(tool, "server_url", None)
        or os.getenv("DOCUWARE_SERVER_URL")
    )
    token = (
        creds.get("token")
        or (tool.token.get_secret_value() if getattr(tool, "token", None) and tool.token else None)
        or os.getenv("DOCUWARE_TOKEN")
    )

    if not server_url:
        raise ValueError(
            "DocuWare server URL not found. Set DOCUWARE_SERVER_URL environment variable, "
            "pass server_url in config, or configure an integration."
        )

    if not token:
        # Fall back to username/password logon
        username = (
            creds.get("username")
            or getattr(tool, "username", None)
            or os.getenv("DOCUWARE_USERNAME")
        )
        password = (
            creds.get("password")
            or (
                tool.password.get_secret_value()
                if getattr(tool, "password", None) and tool.password
                else None
            )
            or os.getenv("DOCUWARE_PASSWORD")
        )
        organization = (
            creds.get("organization")
            or getattr(tool, "organization", None)
            or os.getenv("DOCUWARE_ORGANIZATION")
        )

        if not username or not password:
            raise ValueError(
                "DocuWare credentials not found. Provide a DOCUWARE_TOKEN, or set "
                "DOCUWARE_USERNAME, DOCUWARE_PASSWORD (and optionally DOCUWARE_ORGANIZATION)."
            )

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                _dw_url(server_url, "Account/Logon"),
                json={
                    "UserName": username,
                    "Password": password,
                    "Organization": organization or "",
                    "RememberMe": False,
                    "RedirectToMyselfInCaseOfError": False,
                    "LicenseType": "PlatformService",
                },
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
            response.raise_for_status()
            token = response.cookies.get(".DWPLATFORMAUTH")
            if not token:
                raise ValueError("DocuWare logon succeeded but no auth cookie was returned.")

    return server_url, token


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Cookie": f".DWPLATFORMAUTH={token}",
    }


class DocuWareListFileCabinets(Tool):
    name: str = "docuware_list_file_cabinets"
    description: str | None = "List all file cabinets available in the DocuWare organization."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_file_cabinets() -> Any:
            server_url, token = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _dw_url(server_url, "FileCabinets"),
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_file_cabinets, **kwargs)


class DocuWareSearchDocuments(Tool):
    name: str = "docuware_search_documents"
    description: str | None = "Search for documents in a DocuWare file cabinet using field filters."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_documents(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet to search in"),
            condition: list[dict[str, Any]] = Field(
                ...,
                description=(
                    "List of search conditions. Each condition: "
                    "{'DBName': '<field_name>', 'Value': ['<value>'], 'Condition': 'equal|less|greater|between|isNull'}. "
                    "Example: [{'DBName': 'COMPANY', 'Value': ['Acme Corp'], 'Condition': 'equal'}]"
                ),
            ),
            count: int = Field(50, description="Maximum number of documents to return"),
            start: int = Field(0, description="Offset for pagination"),
            fields: list[str] | None = Field(None, description="List of field names to include in the results"),
            order_field: str | None = Field(None, description="Field name to sort results by"),
            order_descending: bool = Field(False, description="Sort in descending order"),
        ) -> Any:
            server_url, token = await _resolve_credentials(self)
            import httpx

            body: dict[str, Any] = {
                "Condition": condition,
                "Count": count,
                "Start": start,
                "SortOrder": [],
            }
            if fields:
                body["Fields"] = fields
            if order_field:
                body["SortOrder"] = [{"Field": order_field, "Direction": "Desc" if order_descending else "Asc"}]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _dw_url(server_url, f"FileCabinets/{file_cabinet_id}/Query/DialogExpression"),
                    headers={**_auth_headers(token), "Content-Type": "application/json"},
                    json={"Condition": condition, "Count": count, "Start": start, **body},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_documents, **kwargs)


class DocuWareGetDocument(Tool):
    name: str = "docuware_get_document"
    description: str | None = "Retrieve metadata and index fields for a specific DocuWare document."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_document(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet"),
            document_id: int = Field(..., description="The numeric ID of the document"),
            fields: list[str] | None = Field(None, description="List of field names to include in the response"),
        ) -> Any:
            server_url, token = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {}
            if fields:
                params["fields"] = ",".join(fields)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _dw_url(server_url, f"FileCabinets/{file_cabinet_id}/Documents/{document_id}"),
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_document, **kwargs)


class DocuWareDownloadDocument(Tool):
    name: str = "docuware_download_document"
    description: str | None = "Download a DocuWare document as base64-encoded content."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_document(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet"),
            document_id: int = Field(..., description="The numeric ID of the document to download"),
            keep_annotations: bool = Field(False, description="Include annotations in the downloaded PDF"),
        ) -> Any:
            import base64

            server_url, token = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"keepAnnotations": str(keep_annotations).lower()}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _dw_url(
                        server_url,
                        f"FileCabinets/{file_cabinet_id}/Documents/{document_id}/FileDownload",
                    ),
                    headers={**_auth_headers(token), "Accept": "application/pdf,application/octet-stream,*/*"},
                    params=params,
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                return {
                    "content_type": content_type,
                    "content_base64": base64.b64encode(response.content).decode("utf-8"),
                    "size_bytes": len(response.content),
                }

        super().__init__(handler=_download_document, **kwargs)


class DocuWareStoreDocument(Tool):
    name: str = "docuware_store_document"
    description: str | None = "Store (upload) a new document into a DocuWare file cabinet with index fields."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _store_document(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet to store the document in"),
            file_content_base64: str = Field(..., description="Base64-encoded file content to upload"),
            file_name: str = Field(..., description="File name including extension, e.g. 'invoice.pdf'"),
            fields: list[dict[str, Any]] = Field(
                ...,
                description=(
                    "List of index field values. Each entry: {'FieldName': '<name>', 'Item': '<value>'}. "
                    "Example: [{'FieldName': 'COMPANY', 'Item': 'Acme Corp'}, {'FieldName': 'DATE', 'Item': '2026-03-30'}]"
                ),
            ),
            content_type: str = Field("application/pdf", description="MIME type of the file, e.g. 'application/pdf'"),
        ) -> Any:
            import base64
            import json

            server_url, token = await _resolve_credentials(self)
            import httpx

            file_bytes = base64.b64decode(file_content_base64)
            document_json = json.dumps({"Fields": fields})

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _dw_url(server_url, f"FileCabinets/{file_cabinet_id}/Documents"),
                    headers={
                        "Cookie": f".DWPLATFORMAUTH={token}",
                        "Accept": "application/json",
                    },
                    files={
                        "document": (
                            None,
                            document_json,
                            "application/json",
                        ),
                        "file[]": (file_name, file_bytes, content_type),
                    },
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_store_document, **kwargs)


class DocuWareUpdateDocumentFields(Tool):
    name: str = "docuware_update_document_fields"
    description: str | None = "Update index field values of an existing DocuWare document."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_document_fields(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet"),
            document_id: int = Field(..., description="The numeric ID of the document to update"),
            fields: list[dict[str, Any]] = Field(
                ...,
                description=(
                    "List of field values to update. Each entry: {'FieldName': '<name>', 'Item': '<new_value>'}. "
                    "Example: [{'FieldName': 'STATUS', 'Item': 'Approved'}]"
                ),
            ),
        ) -> Any:
            server_url, token = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    _dw_url(server_url, f"FileCabinets/{file_cabinet_id}/Documents/{document_id}/Fields"),
                    headers={**_auth_headers(token), "Content-Type": "application/json"},
                    json={"Field": fields},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_document_fields, **kwargs)


class DocuWareDeleteDocument(Tool):
    name: str = "docuware_delete_document"
    description: str | None = "Delete a document from a DocuWare file cabinet."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_document(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet"),
            document_id: int = Field(..., description="The numeric ID of the document to delete"),
        ) -> Any:
            server_url, token = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _dw_url(server_url, f"FileCabinets/{file_cabinet_id}/Documents/{document_id}"),
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return {"deleted": True, "document_id": document_id}

        super().__init__(handler=_delete_document, **kwargs)


class DocuWareGetFileCabinetFields(Tool):
    name: str = "docuware_get_file_cabinet_fields"
    description: str | None = "Get the index field definitions (schema) for a DocuWare file cabinet."
    integration: Annotated[str, Integration("docuware")] | None = None
    server_url: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    organization: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "server_url": self.server_url,
                "token": self.token,
                "username": self.username,
                "password": self.password,
                "organization": self.organization,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_file_cabinet_fields(
            file_cabinet_id: str = Field(..., description="The GUID of the file cabinet"),
        ) -> Any:
            server_url, token = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _dw_url(server_url, f"FileCabinets/{file_cabinet_id}/Fields"),
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_file_cabinet_fields, **kwargs)
