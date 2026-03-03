import os
from typing import Annotated, Any

import httpx
from pydantic import SecretStr, model_validator

from ..core.tool import Tool
from ..platform.integrations import Integration


class TavilySearch(Tool):
    name: str = "tavily_search"
    description: str | None = (
        "Search the web using Tavily's API with advanced filtering options. "
        "Returns comprehensive search results with answers, raw content, and images."
    )
    integration: Annotated[str, Integration("tavily")] | None = None
    api_key: SecretStr | None = None
    base_url: str = "https://api.tavily.com"

    @model_validator(mode="after")
    def _resolve_credentials(self) -> "TavilySearch":
        if self.integration is None and self.api_key is None:
            env_key = os.getenv("TAVILY_API_KEY")
            if env_key:
                self.api_key = SecretStr(env_key)
        if self.integration is None and not self.api_key:
            raise ValueError(
                "Tavily API key not found. Set TAVILY_API_KEY environment variable, "
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
        async def _tavily_search(
            input: str,
            search_depth: str = "basic",
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
            include_answer: bool = True,
            include_raw_content: bool = True,
            include_images: bool = False,
            max_results: int = 10,
        ) -> dict:
            if self.integration:
                assert isinstance(self.integration, Integration)
                credential = await self.integration.resolve()
                api_key = credential.token
            else:
                assert self.api_key is not None
                api_key = self.api_key.get_secret_value()

            payload = {
                "query": input,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
                "max_results": max_results,
                "include_domains": include_domains if include_domains else None,
                "exclude_domains": exclude_domains if exclude_domains else None
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=httpx.Timeout(10.0, read=None),
                )
            
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Tavily/Search"

        super().__init__(handler=_tavily_search, metadata=metadata, **kwargs)