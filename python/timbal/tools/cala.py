import os
from typing import Annotated, Any

import httpx
from pydantic import SecretStr, model_validator

from ..core.tool import Tool
from ..platform.integrations import Integration


class CalaSearch(Tool):
    name: str = "cala_search"
    description: str | None = (
        "Search for verified knowledge using natural language queries. "
        "Returns trustworthy, verified knowledge with relevant context, sources, and matching entities."
    )
    integration: Annotated[str, Integration("cala")] | None = None
    api_key: SecretStr | None = None
    base_url: str = "https://api.cala.ai/v1"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "CalaSearch":
        if self.integration is None and self.api_key is None:
            env_key = os.getenv("CALA_API_KEY")
            if env_key:
                self.api_key = SecretStr(env_key)
        if self.integration is None and not self.api_key:
            raise ValueError(
                "Cala API key not found. Set CALA_API_KEY environment variable, "
                "pass api_key in config, or configure an integration."
            )
        return self

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
        async def _cala_search(input: str) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                api_key = credential.token
            else:
                assert self.api_key is not None  # Validated in _resolve_credentials
                api_key = self.api_key.get_secret_value()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/knowledge/search",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json={"input": input},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Cala/Search"

        super().__init__(handler=_cala_search, metadata=metadata, **kwargs)
