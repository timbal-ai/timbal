from typing import Any

from httpx import AsyncClient

from ...state.context import TimbalPlatformConfig, run_context_var
from ...types.field import Field


async def get_orgs(
    timbal_platform_config: TimbalPlatformConfig | None = Field(default=None, private=True), 
) -> dict[str, Any]:

    if not isinstance(timbal_platform_config, TimbalPlatformConfig):
        run_context = run_context_var.get(None)
        if run_context is None:
            raise ValueError(
                "RunContext not found. "
                "Please run this function as a timbal step or pass the timbal_platform_config explicitly.")

        timbal_platform_config = run_context.timbal_platform_config
        if not isinstance(timbal_platform_config, TimbalPlatformConfig):
            raise ValueError(
                "Missing Timbal Platform Config. "
                "Please specify it when running this function as a step or pass it explicitly as a function parameter.")

    async with AsyncClient() as client:

        auth = timbal_platform_config.auth
        headers = {auth.header_key: auth.header_value}

        res = await client.get(
            f"https://{timbal_platform_config.host}/me/orgs", 
            headers=headers,
        )
        res.raise_for_status()

        return res.json()
