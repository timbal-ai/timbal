import base64
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_BASE_URL = "https://www.happyscribe.com/api/v1"
_USER_AGENT = "timbal-happy-scribe-tools/1.0"

ExportFormat = Literal[
    "txt",
    "docx",
    "pdf",
    "srt",
    "vtt",
    "stl",
    "avid",
    "avid_script",
    "avid_ds_subtitles",
    "avid_edl",
    "facebook_srt",
    "html",
    "premiere",
    "maxqda",
    "json",
    "fcp",
    "edl",
    "csv",
    "xlsx",
    "mp3",
    "mp4",
]

OrderService = Literal["auto", "pro"]


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }


async def _resolve_organization_id(
    tool: Any,
    organization_id: str | int | None,
    *,
    required: bool = True,
) -> str | None:
    if organization_id is not None:
        return str(organization_id)
    if tool.default_organization_id is not None:
        return str(tool.default_organization_id)
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        org = credentials.get("organization_id")
        if org is not None:
            return str(org)
    env_org = os.getenv("HAPPYSCRIBE_ORGANIZATION_ID")
    if env_org:
        return env_org
    if required:
        raise ValueError(
            "Happy Scribe organization_id is required. Pass organization_id on the call, "
            "set default_organization_id on the tool, store organization_id in the integration, "
            "or set HAPPYSCRIBE_ORGANIZATION_ID."
        )
    return None


def _config_fields(tool: Any) -> dict[str, Any]:
    return tool._annotate_config(
        {
            "integration": tool.integration,
            "api_key": tool.api_key,
            "default_organization_id": tool.default_organization_id,
        }
    )


async def _request(
    tool: Any,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | list[tuple[str, str]] | None = None,
    json_body: Any = None,
    timeout: float = 60.0,
) -> Any:
    api_key = await resolve_api_key(
        env_var="HAPPYSCRIBE_API_KEY",
        provider_name="Happy Scribe",
        integration=tool.integration,
        api_key=tool.api_key,
    )
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.request(
            method,
            f"{_BASE_URL}{path}",
            headers=_headers(api_key),
            params=params,
            json=json_body,
            timeout=httpx.Timeout(timeout),
        )
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {"status": "success"}
        return response.json()


async def _load_upload_bytes(
    *,
    file_path: str | None,
    file_url: str | None,
    file_content_base64: str | None,
) -> tuple[bytes, str | None]:
    if sum(x is not None for x in (file_path, file_url, file_content_base64)) != 1:
        raise ValueError("Provide exactly one of file_path, file_url, or file_content_base64.")
    if file_path is not None:
        path = Path(file_path)
        return path.read_bytes(), path.name
    if file_content_base64 is not None:
        return base64.b64decode(file_content_base64), None
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.get(file_url, timeout=httpx.Timeout(120.0, read=None))
        response.raise_for_status()
        filename = file_url.rsplit("/", 1)[-1].split("?", 1)[0] if file_url else None
        return response.content, filename or None


class _HappyScribeTool(Tool):
    integration: Annotated[str, Integration("happy_scribe")] | None = None
    api_key: SecretStr | None = None
    default_organization_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **_config_fields(self)}


class HappyScribeGetUploadSignedUrl(_HappyScribeTool):
    name: str = "happy_scribe_get_upload_signed_url"
    description: str | None = (
        "Request a signed S3 URL from Happy Scribe for uploading a media file. "
        "Use the returned signedUrl as the media url when creating an order, after PUT-uploading the file."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_signed_url(
            filename: str = Field(..., description="Media filename with extension, e.g. interview.mp3"),
        ) -> Any:
            return await _request(
                self,
                "GET",
                "/uploads/new",
                params={"filename": filename},
            )

        super().__init__(handler=_get_signed_url, **kwargs)


class HappyScribeUploadFile(_HappyScribeTool):
    name: str = "happy_scribe_upload_file"
    description: str | None = (
        "Upload a media file to Happy Scribe storage via a signed URL from get_upload_signed_url. "
        "Returns the signed URL to pass as url when creating an order."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_file(
            signed_url: str = Field(..., description="Signed URL from happy_scribe_get_upload_signed_url"),
            file_path: str | None = Field(None, description="Local path to the media file"),
            file_url: str | None = Field(None, description="Public URL to download the media file from"),
            file_content_base64: str | None = Field(None, description="Base64-encoded media file bytes"),
            content_type: str | None = Field(
                None,
                description="Optional Content-Type for the PUT request (defaults to application/octet-stream)",
            ),
        ) -> Any:
            content, _ = await _load_upload_bytes(
                file_path=file_path,
                file_url=file_url,
                file_content_base64=file_content_base64,
            )
            import httpx

            headers = {"Content-Type": content_type or "application/octet-stream"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.put(
                    signed_url,
                    content=content,
                    headers=headers,
                    timeout=httpx.Timeout(300.0, read=None),
                )
                response.raise_for_status()
            return {"signed_url": signed_url, "uploaded_bytes": len(content)}

        super().__init__(handler=_upload_file, **kwargs)


class HappyScribeListTranscriptions(_HappyScribeTool):
    name: str = "happy_scribe_list_transcriptions"
    description: str | None = (
        "List Happy Scribe transcriptions for an organization. Returns metadata only; use exports to fetch transcript content."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_transcriptions(
            organization_id: str | int | None = Field(
                None,
                description="Happy Scribe workspace/organization ID. Uses tool or integration default when omitted.",
            ),
            folder_id: str | int | None = Field(None, description="Filter to transcriptions in this folder or subfolders"),
            page: int = Field(0, description="Page index (0-based)"),
            per_page: int = Field(5, description="Results per page (max 100)"),
            tags: list[str] | None = Field(None, description="Filter by tags"),
        ) -> Any:
            org_id = await _resolve_organization_id(self, organization_id)
            params: list[tuple[str, str]] = [("organization_id", org_id)]
            if folder_id is not None:
                params.append(("folder_id", str(folder_id)))
            params.append(("page", str(page)))
            params.append(("per_page", str(per_page)))
            if tags:
                for tag in tags:
                    params.append(("tags[]", tag))
            return await _request(self, "GET", "/transcriptions", params=params)

        super().__init__(handler=_list_transcriptions, **kwargs)


class HappyScribeGetTranscription(_HappyScribeTool):
    name: str = "happy_scribe_get_transcription"
    description: str | None = (
        "Retrieve Happy Scribe transcription metadata and processing state. Use exports to download transcript content."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_transcription(
            transcription_id: str = Field(..., description="Happy Scribe transcription ID"),
        ) -> Any:
            return await _request(self, "GET", f"/transcriptions/{transcription_id}")

        super().__init__(handler=_get_transcription, **kwargs)


class HappyScribeDeleteTranscription(_HappyScribeTool):
    name: str = "happy_scribe_delete_transcription"
    description: str | None = "Delete a Happy Scribe transcription (moves to Trash unless permanent=true)."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_transcription(
            transcription_id: str = Field(..., description="Happy Scribe transcription ID"),
            permanent: bool = Field(False, description="Irreversibly delete when true"),
        ) -> Any:
            params = {"permanent": "true"} if permanent else None
            return await _request(self, "DELETE", f"/transcriptions/{transcription_id}", params=params)

        super().__init__(handler=_delete_transcription, **kwargs)


class HappyScribeCreateExport(_HappyScribeTool):
    name: str = "happy_scribe_create_export"
    description: str | None = "Export one or more Happy Scribe transcriptions to txt, srt, json, docx, and other formats."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_export(
            format: ExportFormat = Field(..., description="Export format"),
            transcription_ids: list[str] = Field(..., description="Transcription IDs to export"),
            show_timestamps: bool = Field(False, description="Include timestamps (txt, docx, pdf)"),
            timestamps_frequency: Literal["5s", "10s", "15s", "20s", "30s", "60s"] | None = Field(
                None,
                description="Inline timecode frequency",
            ),
            show_speakers: bool = Field(False, description="Include speaker labels (txt, docx, pdf)"),
            show_comments: bool = Field(False, description="Include comments (txt, docx, pdf)"),
            show_highlights: bool = Field(False, description="Include highlights (docx, pdf)"),
            show_highlights_only: bool = Field(False, description="Export highlights only (docx, pdf)"),
        ) -> Any:
            export: dict[str, Any] = {
                "format": format,
                "transcription_ids": transcription_ids,
                "show_timestamps": show_timestamps,
                "show_speakers": show_speakers,
                "show_comments": show_comments,
                "show_highlights": show_highlights,
                "show_highlights_only": show_highlights_only,
            }
            if timestamps_frequency is not None:
                export["timestamps_frequency"] = timestamps_frequency
            return await _request(self, "POST", "/exports", json_body={"export": export})

        super().__init__(handler=_create_export, **kwargs)


class HappyScribeGetExport(_HappyScribeTool):
    name: str = "happy_scribe_get_export"
    description: str | None = "Retrieve a Happy Scribe export job status and download_link when ready."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_export(
            export_id: str = Field(..., description="Export ID from create_export"),
        ) -> Any:
            return await _request(self, "GET", f"/exports/{export_id}")

        super().__init__(handler=_get_export, **kwargs)


class HappyScribeCreateOrder(_HappyScribeTool):
    name: str = "happy_scribe_create_order"
    description: str | None = (
        "Create a Happy Scribe transcription or subtitling order from a media URL "
        "(public link, YouTube/Vimeo, or signed upload URL)."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_order(
            url: str = Field(..., description="Publicly accessible media file URL"),
            language: str = Field(..., description="BCP-47 language code"),
            organization_id: str | int | None = Field(None, description="Workspace ID; uses default when omitted"),
            service: OrderService = Field("auto", description="auto or pro (human)"),
            confirm: bool = Field(False, description="Submit immediately when true"),
            folder_id: str | int | None = Field(None, description="Target folder ID"),
            folder: str | None = Field(None, description="Target folder path"),
            name: str | None = Field(None, description="Resulting transcription name"),
            boost: bool = Field(False, description="Boost order priority"),
            is_subtitle: bool = Field(False, description="Create subtitles instead of transcription"),
            tags: list[str] | None = Field(None, description="Tags for the resulting transcription"),
            webhook_url: str | None = Field(None, description="Webhook URL for order state changes"),
            glossary_ids: list[int] | None = Field(None, description="Glossary IDs from list_glossaries"),
            style_guide_id: int | None = Field(None, description="Style guide ID from list_style_guides"),
        ) -> Any:
            org_id = await _resolve_organization_id(self, organization_id)
            order: dict[str, Any] = {
                "url": url,
                "language": language,
                "organization_id": int(org_id) if org_id.isdigit() else org_id,
                "service": service,
                "confirm": confirm,
                "boost": boost,
                "is_subtitle": is_subtitle,
            }
            if folder_id is not None:
                order["folder_id"] = int(folder_id) if str(folder_id).isdigit() else folder_id
            if folder is not None:
                order["folder"] = folder
            if name is not None:
                order["name"] = name
            if tags:
                order["tags"] = tags
            if webhook_url is not None:
                order["webhook_url"] = webhook_url
            if glossary_ids:
                order["glossary_ids"] = glossary_ids
            if style_guide_id is not None:
                order["style_guide_id"] = style_guide_id
            return await _request(self, "POST", "/orders", json_body={"order": order})

        super().__init__(handler=_create_order, **kwargs)


class HappyScribeCreateTranslationOrder(_HappyScribeTool):
    name: str = "happy_scribe_create_translation_order"
    description: str | None = "Create a translation order from an existing transcription into one or more target languages."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_translation_order(
            source_transcription_id: str = Field(..., description="Source transcription ID"),
            target_languages: list[str] = Field(..., description="Target language codes, e.g. ['es', 'fr']"),
            service: OrderService = Field("auto", description="auto or pro (human)"),
            confirm: bool = Field(False, description="Submit immediately when true"),
            boost: bool = Field(False, description="Boost order priority"),
            webhook_url: str | None = Field(None, description="Webhook URL for order state changes"),
            tags: list[str] | None = Field(None, description="Tags for resulting transcriptions"),
        ) -> Any:
            order: dict[str, Any] = {
                "source_transcription_id": source_transcription_id,
                "target_languages": target_languages,
                "service": service,
                "confirm": confirm,
                "boost": boost,
            }
            if webhook_url is not None:
                order["webhook_url"] = webhook_url
            if tags:
                order["tags"] = tags
            return await _request(self, "POST", "/orders/translation", json_body={"order": order})

        super().__init__(handler=_create_translation_order, **kwargs)


class HappyScribeGetOrder(_HappyScribeTool):
    name: str = "happy_scribe_get_order"
    description: str | None = "Retrieve a Happy Scribe order status, pricing details, and output transcription IDs."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_order(
            order_id: str = Field(..., description="Order ID"),
        ) -> Any:
            return await _request(self, "GET", f"/orders/{order_id}")

        super().__init__(handler=_get_order, **kwargs)


class HappyScribeConfirmOrder(_HappyScribeTool):
    name: str = "happy_scribe_confirm_order"
    description: str | None = "Confirm an order created with confirm=false so processing begins."

    def __init__(self, **kwargs: Any) -> None:
        async def _confirm_order(
            order_id: str = Field(..., description="Order ID to confirm"),
        ) -> Any:
            return await _request(self, "POST", f"/orders/{order_id}/confirm")

        super().__init__(handler=_confirm_order, **kwargs)


class HappyScribeListGlossaries(_HappyScribeTool):
    name: str = "happy_scribe_list_glossaries"
    description: str | None = "List glossaries in a Happy Scribe organization for use when creating orders."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_glossaries(
            organization_id: str | int | None = Field(None, description="Workspace ID; uses default when omitted"),
        ) -> Any:
            org_id = await _resolve_organization_id(self, organization_id)
            return await _request(
                self,
                "GET",
                "/glossaries",
                params={"organization_id": org_id},
            )

        super().__init__(handler=_list_glossaries, **kwargs)


class HappyScribeListStyleGuides(_HappyScribeTool):
    name: str = "happy_scribe_list_style_guides"
    description: str | None = "List style guides in a Happy Scribe organization for use when creating pro orders."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_style_guides(
            organization_id: str | int | None = Field(None, description="Workspace ID; uses default when omitted"),
        ) -> Any:
            org_id = await _resolve_organization_id(self, organization_id)
            return await _request(
                self,
                "GET",
                "/style_guides",
                params={"organization_id": org_id},
            )

        super().__init__(handler=_list_style_guides, **kwargs)
