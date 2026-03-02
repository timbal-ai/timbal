import base64
from typing import Annotated, Any

import httpx
from pydantic import BaseModel

from ..core.tool import Tool
from ..platform.integrations import Integration


class GmailConfig(BaseModel):
    integration: Annotated[str, Integration("gmail")]


class SendEmail(Tool):
    config: GmailConfig

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_email(to: str, subject: str, body: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            message = f"To: {to}\r\nSubject: {subject}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}"
            raw = base64.urlsafe_b64encode(message.encode()).decode()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"raw": raw},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Gmail/SendEmail"

        super().__init__(
            name="gmail_send_email",
            description="Send an email using Gmail.",
            handler=_send_email,
            metadata=metadata,
            **kwargs,
        )
