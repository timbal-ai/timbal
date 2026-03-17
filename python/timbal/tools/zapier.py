import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://actions.zapier.com"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Zapier AI Actions API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("ZAPIER_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Zapier API key not found. Set ZAPIER_API_KEY or configure an integration. "
        "Get your key from https://actions.zapier.com/credentials/"
    )


class ZapierCheckAuth(Tool):
    name: str = "zapier_check_auth"
    description: str | None = "Verify Zapier API key and get connected user info."
    integration: Annotated[str, Integration("zapier")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _check_auth() -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/api/v2/auth/check/",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_check_auth, **kwargs)


class ZapierTriggerWebhook(Tool):
    name: str = "zapier_trigger_webhook"
    description: str | None = "Trigger a Zap via its webhook URL (Webhooks by Zapier, Premium)."
    integration: Annotated[str, Integration("zapier")] | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _trigger_webhook(
            webhook_url: str = Field(
                ...,
                description="Webhook URL from your Zap (Webhooks by Zapier → Catch Hook).",
            ),
            payload: dict[str, Any] = Field(
                default_factory=dict,
                description="JSON payload to send to the webhook.",
            ),
        ) -> Any:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                try:
                    return response.json()
                except Exception:
                    return {"status": response.status_code, "body": response.text}

        super().__init__(handler=_trigger_webhook, **kwargs)
