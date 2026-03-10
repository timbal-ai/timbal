from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Tavily API key from integration or api_key field."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    elif tool.api_key:
        return tool.api_key.get_secret_value()
    raise ValueError("Tavily integration not configured and no API key provided.")


class TavilySearch(Tool):
    name: str = "tavily_search"
    description: str | None = (
        "Search the web using Tavily's API with advanced filtering options. "
        "Returns comprehensive search results with answers, raw content, and images."
    )
    integration: Annotated[str, Integration("tavily")] | None = None
    api_key: SecretStr | None = None
    base_url: str = "https://api.tavily.com"

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
            input: str = Field(..., description="Search query"),
            search_depth: str = Field("basic", description="Search depth: 'basic' or 'advanced'"),
            include_domains: list[str] | None = Field(None, description="Domains to include in search"),
            exclude_domains: list[str] | None = Field(None, description="Domains to exclude from search"),
            include_answer: bool = Field(True, description="Include answer in results"),
            include_raw_content: bool = Field(True, description="Include raw content in results"),
            include_images: bool = Field(False, description="Include images in results"),
            max_results: int = Field(10, description="Maximum number of results to return"),
        ) -> dict:
            api_key = await _resolve_api_key(self)

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

            import httpx

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