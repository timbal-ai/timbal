import os
from typing import Annotated, Any

<<<<<<< HEAD
from pydantic import SecretStr, model_validator
=======
from pydantic import Field, SecretStr, model_validator
>>>>>>> 1ae72bff7900fdebd9db6e3666ec946fcd888a32

from ..core.tool import Tool
from ..platform.integrations import Integration


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Cala API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("CALA_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Cala API key not found. Set CALA_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class CalaSearch(Tool):
    name: str = "cala_search"
    description: str | None = (
        "Search for verified knowledge using natural language queries. "
        "Returns trustworthy, verified knowledge with relevant context, sources, and matching entities."
    )
    integration: Annotated[str, Integration("cala")] | None = None
    api_key: SecretStr | None = None
    base_url: str = "https://api.cala.ai/v1"

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
        async def _cala_search(input: str = Field(..., description="Search query for Cala API")) -> dict:
            api_key = await _resolve_api_key(self)

            import httpx

            import httpx

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
