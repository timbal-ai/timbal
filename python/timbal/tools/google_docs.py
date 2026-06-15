from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_DOCS_BASE = "https://docs.googleapis.com/v1"
_DRIVE_BASE = "https://www.googleapis.com/drive/v3"
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError(
        "Google Docs credentials not found. Configure an integration or pass token."
    )


def _config_fields(tool: Any) -> dict[str, Any]:
    return tool._annotate_config({"integration": tool.integration, "token": tool.token})


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _location(*, index: int, tab_id: str | None = None) -> dict[str, Any]:
    loc: dict[str, Any] = {"index": index}
    if tab_id:
        loc["tabId"] = tab_id
    return loc


def _body_content(document: dict[str, Any]) -> list[dict[str, Any]]:
    return document.get("body", {}).get("content", [])


def _document_end_index(document: dict[str, Any], *, tab_id: str | None = None) -> int:
    if tab_id:
        tab = _find_tab(document, tab_id)
        if tab is None:
            raise ValueError(f"Tab not found: {tab_id}")
        content = tab.get("documentTab", {}).get("body", {}).get("content", [])
    else:
        content = _body_content(document)
    if not content:
        return 1
    return content[-1]["endIndex"] - 1


def _find_tab(document: dict[str, Any], tab_id: str) -> dict[str, Any] | None:
    for tab in document.get("tabs", []):
        props = tab.get("tabProperties", {})
        if props.get("tabId") == tab_id:
            return tab
        child = tab.get("childTabs") or []
        for nested in child:
            nested_props = nested.get("tabProperties", {})
            if nested_props.get("tabId") == tab_id:
                return nested
    return None


def _find_tab_by_title(document: dict[str, Any], tab_title: str) -> dict[str, Any] | None:
    for tab in document.get("tabs", []):
        props = tab.get("tabProperties", {})
        if props.get("title") == tab_title:
            return tab
    return None


async def _get_document(
    client: Any,
    token: str,
    document_id: str,
    *,
    include_tabs_content: bool = False,
) -> dict[str, Any]:
    params: dict[str, str] = {}
    if include_tabs_content:
        params["includeTabsContent"] = "true"
    response = await client.get(
        f"{_DOCS_BASE}/documents/{document_id}",
        headers=_auth_headers(token),
        params=params or None,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


async def _batch_update(
    client: Any,
    token: str,
    document_id: str,
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    response = await client.post(
        f"{_DOCS_BASE}/documents/{document_id}:batchUpdate",
        headers=_auth_headers(token),
        json={"requests": requests},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


async def _create_document_file(
    client: Any,
    token: str,
    *,
    title: str,
    folder_id: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": title, "mimeType": _GOOGLE_DOC_MIME}
    if folder_id:
        metadata["parents"] = [folder_id]
    response = await client.post(
        f"{_DRIVE_BASE}/files",
        headers=_auth_headers(token),
        json=metadata,
        params={"fields": "id,name,webViewLink"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


class GoogleDocsFindDocument(Tool):
    name: str = "google_docs_find_document"
    description: str | None = (
        "Search for a Google Docs file by name using Drive full-text tokenized matching. "
        "Pass a distinctive word or short phrase rather than the full title when the name "
        "contains special characters like & or '."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _find_document(
            search_name: str = Field(
                ...,
                description=(
                    "Distinctive word or phrase from the document title. Uses Drive fullText "
                    "tokenized matching; prefer a short unique term over the full title."
                ),
            ),
            folder_id: str | None = Field(None, description="Optional parent folder ID to limit search"),
            max_results: int = Field(10, description="Maximum number of matching documents to return"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            escaped = _escape_drive_query(search_name)
            query_parts = [
                f"mimeType='{_GOOGLE_DOC_MIME}'",
                f"fullText contains '{escaped}'",
                "trashed=false",
            ]
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_DRIVE_BASE}/files",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "q": " and ".join(query_parts),
                        "pageSize": max_results,
                        "fields": "files(id,name,webViewLink,modifiedTime,createdTime,parents)",
                        "supportsAllDrives": "true",
                        "includeItemsFromAllDrives": "true",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

            files = [
                {
                    "documentId": item["id"],
                    "title": item["name"],
                    "documentUrl": item.get("webViewLink"),
                    "modifiedTime": item.get("modifiedTime"),
                    "createdTime": item.get("createdTime"),
                    "parentIds": item.get("parents", []),
                }
                for item in data.get("files", [])
            ]
            return {"documents": files, "total": len(files)}

        super().__init__(handler=_find_document, **kwargs)


class GoogleDocsCreateFromTemplate(Tool):
    name: str = "google_docs_create_from_template"
    description: str | None = (
        "Create a new Google Doc by copying a template and replacing placeholder text. "
        "Placeholders in the template are replaced via replaceAllText (plain text)."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _create_from_template(
            template_id: str = Field(..., description="Google Docs template file ID"),
            title: str = Field(..., description="Title for the new document"),
            replacements: dict[str, str] = Field(
                default_factory=dict,
                description='Placeholder text to replace, e.g. {"{{name}}": "Jane", "{{date}}": "2026-06-05"}',
            ),
            folder_id: str | None = Field(None, description="Optional folder ID for the new document"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            copy_body: dict[str, Any] = {"name": title}
            if folder_id:
                copy_body["parents"] = [folder_id]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DRIVE_BASE}/files/{template_id}/copy",
                    headers=_auth_headers(token),
                    json=copy_body,
                    params={"fields": "id,name,webViewLink"},
                    timeout=30.0,
                )
                response.raise_for_status()
                copied = response.json()
                document_id = copied["id"]

                if replacements:
                    requests = [
                        {
                            "replaceAllText": {
                                "containsText": {"text": find_text, "matchCase": True},
                                "replaceText": replace_text,
                            }
                        }
                        for find_text, replace_text in replacements.items()
                    ]
                    await _batch_update(client, token, document_id, requests)

            return {
                "documentId": document_id,
                "title": copied.get("name", title),
                "documentUrl": copied.get("webViewLink") or f"https://docs.google.com/document/d/{document_id}",
            }

        super().__init__(handler=_create_from_template, **kwargs)


class GoogleDocsReplaceText(Tool):
    name: str = "google_docs_replace_text"
    description: str | None = (
        "Replace all instances of matched text in a Google Doc using replaceAllText. "
        "Replacement text is inserted as plain text."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _replace_text(
            document_id: str = Field(..., description="Google Docs document ID"),
            find_text: str = Field(..., description="Text to find in the document"),
            replace_text: str = Field(..., description="Replacement text"),
            match_case: bool = Field(True, description="Whether the search is case-sensitive"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            request = {
                "replaceAllText": {
                    "containsText": {"text": find_text, "matchCase": match_case},
                    "replaceText": replace_text,
                }
            }

            async with httpx.AsyncClient() as client:
                result = await _batch_update(client, token, document_id, [request])

            replies = result.get("replies", [])
            occurrences = 0
            if replies and "replaceAllText" in replies[0]:
                occurrences = replies[0]["replaceAllText"].get("occurrencesChanged", 0)
            return {"documentId": document_id, "occurrencesChanged": occurrences}

        super().__init__(handler=_replace_text, **kwargs)


class GoogleDocsReplaceImage(Tool):
    name: str = "google_docs_replace_image"
    description: str | None = "Replace an inline image in a Google Doc by image object ID."
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _replace_image(
            document_id: str = Field(..., description="Google Docs document ID"),
            image_object_id: str = Field(..., description="Inline image object ID from the document"),
            image_uri: str = Field(..., description="Public HTTPS URL of the replacement image"),
            image_replace_method: str = Field(
                "CENTER_CROP",
                description='Image sizing method: "CENTER_CROP" or "CENTER_INSIDE"',
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            request = {
                "replaceImage": {
                    "imageObjectId": image_object_id,
                    "uri": image_uri,
                    "imageReplaceMethod": image_replace_method,
                }
            }

            async with httpx.AsyncClient() as client:
                await _batch_update(client, token, document_id, [request])

            return {"documentId": document_id, "imageObjectId": image_object_id}

        super().__init__(handler=_replace_image, **kwargs)


class GoogleDocsInsertText(Tool):
    name: str = "google_docs_insert_text"
    description: str | None = "Insert text at a specific index in a Google Doc (optionally in a tab)."
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_text(
            document_id: str = Field(..., description="Google Docs document ID"),
            text: str = Field(..., description="Text to insert"),
            index: int = Field(1, description="Character index where text is inserted (1-based)"),
            tab_id: str | None = Field(None, description="Optional tab ID for multi-tab documents"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            request = {
                "insertText": {
                    "location": _location(index=index, tab_id=tab_id),
                    "text": text,
                }
            }

            async with httpx.AsyncClient() as client:
                await _batch_update(client, token, document_id, [request])

            return {"documentId": document_id, "insertedLength": len(text)}

        super().__init__(handler=_insert_text, **kwargs)


class GoogleDocsInsertTable(Tool):
    name: str = "google_docs_insert_table"
    description: str | None = "Insert a table into a Google Doc at a specific index or end of a tab."
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_table(
            document_id: str = Field(..., description="Google Docs document ID"),
            rows: int = Field(..., description="Number of table rows"),
            columns: int = Field(..., description="Number of table columns"),
            index: int | None = Field(None, description="Insert index (1-based). Omit to append at end of tab."),
            tab_id: str | None = Field(None, description="Optional tab ID for multi-tab documents"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                if index is None:
                    doc = await _get_document(client, token, document_id, include_tabs_content=tab_id is not None)
                    index = _document_end_index(doc, tab_id=tab_id)

                request: dict[str, Any] = {
                    "insertTable": {
                        "rows": rows,
                        "columns": columns,
                        "location": _location(index=index, tab_id=tab_id),
                    }
                }
                result = await _batch_update(client, token, document_id, [request])

            table_id = None
            replies = result.get("replies", [])
            if replies and "insertTable" in replies[0]:
                table_id = replies[0]["insertTable"].get("tableId")
            return {"documentId": document_id, "tableId": table_id, "rows": rows, "columns": columns}

        super().__init__(handler=_insert_table, **kwargs)


class GoogleDocsInsertPageBreak(Tool):
    name: str = "google_docs_insert_page_break"
    description: str | None = "Insert a page break into a Google Doc at a specific index."
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_page_break(
            document_id: str = Field(..., description="Google Docs document ID"),
            index: int | None = Field(None, description="Insert index (1-based). Omit to append at document end."),
            tab_id: str | None = Field(None, description="Optional tab ID for multi-tab documents"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                if index is None:
                    doc = await _get_document(client, token, document_id, include_tabs_content=tab_id is not None)
                    index = _document_end_index(doc, tab_id=tab_id)

                request = {
                    "insertPageBreak": {
                        "location": _location(index=index, tab_id=tab_id),
                    }
                }
                await _batch_update(client, token, document_id, [request])

            return {"documentId": document_id, "index": index}

        super().__init__(handler=_insert_page_break, **kwargs)


class GoogleDocsGetTabContent(Tool):
    name: str = "google_docs_get_tab_content"
    description: str | None = (
        "Get the content of a specific tab in a multi-tab Google Doc. "
        "Identify the tab by tab_id or tab_title."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_tab_content(
            document_id: str = Field(..., description="Google Docs document ID"),
            tab_id: str | None = Field(None, description="Tab ID"),
            tab_title: str | None = Field(None, description="Tab title (used if tab_id is not provided)"),
        ) -> dict:
            if not tab_id and not tab_title:
                raise ValueError("Provide tab_id or tab_title.")

            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                document = await _get_document(client, token, document_id, include_tabs_content=True)

            tab = _find_tab(document, tab_id) if tab_id else _find_tab_by_title(document, tab_title or "")
            if tab is None:
                raise ValueError("Tab not found.")

            tab_props = tab.get("tabProperties", {})
            document_tab = tab.get("documentTab", {})
            return {
                "documentId": document_id,
                "tabId": tab_props.get("tabId"),
                "tabTitle": tab_props.get("title"),
                "body": document_tab.get("body"),
                "headers": document_tab.get("headers"),
                "footers": document_tab.get("footers"),
            }

        super().__init__(handler=_get_tab_content, **kwargs)


class GoogleDocsGetDocument(Tool):
    name: str = "google_docs_get_document"
    description: str | None = (
        "Get the contents of the latest version of a Google Doc. "
        "Set include_tabs_content=true for multi-tab documents."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _get_document_tool(
            document_id: str = Field(..., description="Google Docs document ID"),
            include_tabs_content: bool = Field(
                False,
                description="Include full content for all tabs (required for multi-tab documents)",
            ),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                return await _get_document(
                    client,
                    token,
                    document_id,
                    include_tabs_content=include_tabs_content,
                )

        super().__init__(handler=_get_document_tool, **kwargs)


class GoogleDocsCreate(Tool):
    name: str = "google_docs_create"
    description: str | None = (
        "Create a new Google Docs document with optional initial content and folder placement."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _create_document(
            title: str = Field(..., description="Title of the document to create"),
            content: str | None = Field(None, description="Initial plain text content"),
            folder_id: str | None = Field(None, description="Optional Google Drive folder ID"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                created = await _create_document_file(client, token, title=title, folder_id=folder_id)
                document_id = created["id"]

                if content:
                    await _batch_update(
                        client,
                        token,
                        document_id,
                        [{"insertText": {"location": {"index": 1}, "text": content}}],
                    )

            return {
                "documentId": document_id,
                "documentUrl": created.get("webViewLink") or f"https://docs.google.com/document/d/{document_id}",
                "title": created.get("name", title),
            }

        super().__init__(handler=_create_document, **kwargs)


class GoogleDocsAppendText(Tool):
    name: str = "google_docs_append_text"
    description: str | None = "Append plain text to the end of a Google Doc (optionally in a specific tab)."
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _append_text(
            document_id: str = Field(..., description="Google Docs document ID"),
            text: str = Field(..., description="Text to append"),
            tab_id: str | None = Field(None, description="Optional tab ID for multi-tab documents"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                document = await _get_document(client, token, document_id, include_tabs_content=tab_id is not None)
                index = _document_end_index(document, tab_id=tab_id)
                await _batch_update(
                    client,
                    token,
                    document_id,
                    [{"insertText": {"location": _location(index=index, tab_id=tab_id), "text": text}}],
                )

            return {"documentId": document_id, "appendedLength": len(text)}

        super().__init__(handler=_append_text, **kwargs)


class GoogleDocsAppendImage(Tool):
    name: str = "google_docs_append_image"
    description: str | None = (
        "Append an inline image to the end of a Google Doc. "
        "The image_uri must be publicly accessible over HTTPS."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _append_image(
            document_id: str = Field(..., description="Google Docs document ID"),
            image_uri: str = Field(..., description="Public HTTPS URL of the image"),
            width_pt: float = Field(200.0, description="Image width in points"),
            height_pt: float = Field(200.0, description="Image height in points"),
            tab_id: str | None = Field(None, description="Optional tab ID for multi-tab documents"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                document = await _get_document(client, token, document_id, include_tabs_content=tab_id is not None)
                index = _document_end_index(document, tab_id=tab_id)
                request = {
                    "insertInlineImage": {
                        "uri": image_uri,
                        "location": _location(index=index, tab_id=tab_id),
                        "objectSize": {
                            "width": {"magnitude": width_pt, "unit": "PT"},
                            "height": {"magnitude": height_pt, "unit": "PT"},
                        },
                    }
                }
                result = await _batch_update(client, token, document_id, [request])

            object_id = None
            replies = result.get("replies", [])
            if replies and "insertInlineImage" in replies[0]:
                object_id = replies[0]["insertInlineImage"].get("objectId")
            return {"documentId": document_id, "objectId": object_id}

        super().__init__(handler=_append_image, **kwargs)
