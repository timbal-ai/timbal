from typing import Annotated, Any

import httpx
from pydantic import model_validator

from ..core.tool import Tool
from ..platform.integrations import Integration


class GoogleDocsCreate(Tool):
    name: str = "google_docs_create"
    description: str | None = (
        "Create a new Google Docs document with the specified title and content. "
        "Returns document ID and URL for accessing the created document."
    )
    integration: Annotated[str, Integration("google_docs")] | None = None
    base_url: str = "https://docs.googleapis.com/v1"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "GoogleDocsCreate":
        if self.integration is None:
            raise ValueError(
                "Google Docs integration not found. Please configure the google_docs integration."
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
        async def _create_document(
            title: str,
            content: str | None = None,
            folder_id: str | None = None,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                token = credential.token

            document_body = {"title": title}
            
            if folder_id:
                document_body["parents"] = [folder_id]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/documents",
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
                        f"{self.base_url}/documents/{document_id}:batchUpdate",
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GoogleDocs/Create"

        super().__init__(handler=_create_document, metadata=metadata, **kwargs)
