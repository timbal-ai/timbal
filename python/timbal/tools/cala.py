import os
from typing import Annotated, Any

import httpx
from pydantic import BaseModel, Field

from ..core.tool import Tool
from ..platform.integrations import Integration


class CalaConfig(BaseModel):
    integration: Annotated[str, Integration("cala")] | None = None
    api_key: str | None = None
    base_url: str = "https://api.cala.ai/v1"

    def model_post_init(self, __context):
        if self.integration is None and self.api_key is None:
            self.api_key = os.getenv("CALA_API_KEY")
        if self.integration is None and not self.api_key:
            raise ValueError(
                "Cala API key not found. Set CALA_API_KEY environment variable, "
                "pass api_key in config, or configure an integration."
            )


class CalaSearch(Tool):
    config: CalaConfig = Field(default_factory=CalaConfig)

    def __init__(self, **kwargs: Any) -> None:
        async def _cala_search(input: str) -> dict:
            if self.config.integration:
                assert isinstance(self.config.integration, Integration)
                credential = await self.config.integration.resolve()
                api_key = credential.token
            else:
                assert self.config.api_key is not None  # Validated in CalaConfig.model_post_init
                api_key = self.config.api_key

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.base_url}/knowledge/search",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json={"input": input},
                    timeout=httpx.Timeout(10.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Cala/Search"

        super().__init__(
            name="cala_search",
            description=(
                "Search for verified knowledge using natural language queries. "
                "Returns trustworthy, verified knowledge with relevant context, sources, and matching entities."
            ),
            handler=_cala_search,
            metadata=metadata,
            **kwargs,
        )
