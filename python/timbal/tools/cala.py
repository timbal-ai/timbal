import os
from typing import Any

import httpx

from ..core.tool import Tool


class CalaSearch(Tool):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.cala.ai/v1",
        timeout: httpx.Timeout = httpx.Timeout(300, connect=10),
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("CALA_API_KEY")
        if not resolved_api_key:
            raise ValueError("Cala API key not found. Set CALA_API_KEY environment variable or pass api_key parameter.")

        async def _cala_search(input: str) -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/knowledge/search",
                    headers={"x-api-key": resolved_api_key, "Content-Type": "application/json"},
                    json={"input": input},
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(
            name="cala_search",
            description=(
                "Search for verified knowledge using natural language queries. "
                "Returns trustworthy, verified knowledge with relevant context, sources, and matching entities."
            ),
            handler=_cala_search,
            **kwargs,
        )
