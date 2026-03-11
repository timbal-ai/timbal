import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Google Docs API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("GOOGLE_DOCS_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Google Docs API key not found. Set GOOGLE_DOCS_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class GoogleDocsCreate(Tool):
    name: str = "google_docs_create"
    description: str | None = (
        "Create a new Google Docs document with specified title and content. "
        "Returns document ID and URL for accessing the created document."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    api_key: SecretStr | None = None
    base_url: str = "https://docs.googleapis.com/v1"

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_document(
            title: str = Field(..., description="Title of the document to create"),
            content: str | None = Field(None, description="Content of the document to create"),
            folder_id: str | None = Field(None, description="ID of the folder to create the document in (optional)"),
        ) -> dict:
            api_key = await _resolve_api_key(self)
            import httpx

            document_body = {"title": title}
            
            if folder_id:
                document_body["parents"] = [folder_id]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/documents",
                    headers={
                        "Authorization": f"Bearer {api_key}",
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
                        f"{self.base_url}/documents/{document_id}:batchUpdate",
                        headers={
                            "Authorization": f"Bearer {api_key}",
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDocs/Create"

        super().__init__(handler=_create_document, metadata=metadata, **kwargs)
