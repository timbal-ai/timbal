from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_DOCS_BASE = "https://docs.googleapis.com/v1"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError(
        "Google Docs credentials not found. Configure an integration or pass token."
    )


class GoogleDocsCreate(Tool):
    name: str = "google_docs_create"
    description: str | None = (
        "Create a new Google Docs document with specified title and content. "
        "Returns document ID and URL for accessing the created document."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_document(
            title: str = Field(..., description="Title of the document to create"),
            content: str | None = Field(None, description="Content of the document to create"),
            folder_id: str | None = Field(None, description="ID of the folder to create the document in (optional)"),
        ) -> dict:
            token = await _resolve_token(self)
            import httpx

            document_body = {"title": title}
            
            if folder_id:
                document_body["parents"] = [folder_id]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DOCS_BASE}/documents",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json=document_body,
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                document_data = response.json()
                
                document_id = document_data["documentId"]
                if content:
                    insert_request = {
                        "requests": [
                            {
                                "insertText": {
                                    "location": {
                                        "index": 1
                                    },
                                    "text": content
                                }
                            }
                        ]
                    }
                    
                    response = await client.post(
                        f"{_DOCS_BASE}/documents/{document_id}:batchUpdate",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        },
                        json=insert_request,
                        timeout=httpx.Timeout(10.0, read=None),
                    )
                    response.raise_for_status()
                
                return {
                    "documentId": document_id,
                    "documentUrl": f"https://docs.google.com/document/d/{document_id}",
                    "title": title
                }

        super().__init__(handler=_create_document, **kwargs)
