import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.cala.ai/v1"


async def _resolve_api_key(*, integration: Any = None, api_key: SecretStr | None = None) -> str:
    """Resolve Cala API key from integration, explicit field, or env var."""
    if isinstance(integration, Integration):
        credentials = await integration.resolve()
        return credentials["api_key"]
    if api_key is not None:
        return api_key.get_secret_value()
    env_key = os.getenv("CALA_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Cala API key not found. Set CALA_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class CalaSearch(Tool):
    name: str = "cala_search"
    description: str | None = "Search for verified knowledge using natural language queries."
    integration: Annotated[str, Integration("cala")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cala_search(
            query: str = Field(..., description="Natural language search query"),
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/knowledge/search",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json={"input": query},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_cala_search, **kwargs)
