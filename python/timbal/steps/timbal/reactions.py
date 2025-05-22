from typing import Literal

from httpx import AsyncClient

from ...state.context import RunContext, run_context_var
from ...types.field import Field


async def post_reaction(
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the reaction",
    ),
    feedback: str = Field(
        default=None, 
        description="Optional additional feedback",
    ),
    run_context: RunContext | None = Field(default=None, private=True), 
) -> None:

    if not isinstance(run_context, RunContext):
        run_context = run_context_var.get(None)
        if run_context is None:
            raise ValueError(
                "RunContext not found. "
                "Please run this function as a timbal step or pass the run_context explicitly.")

    timbal_platform_config = run_context.timbal_platform_config
    if not timbal_platform_config:
        raise ValueError(
            "Missing Timbal Platform Config. "
            "Please specify it when running this function as a step or pass it explicitly as a function parameter.")

    # Platform scope.
    org_id = timbal_platform_config.scope.org_id
    app_id = timbal_platform_config.scope.app_id
    run_id = run_context.id
    if org_id is None or app_id is None or run_id is None:
        raise ValueError(
            "Missing org_id or app_id or run_id. "
            "Please specify it when running this function as a step or pass it explicitly as a function parameter.")

    async with AsyncClient() as client:

        auth = timbal_platform_config.auth
        headers = {auth.header_key: auth.header_value}

        payload = {
            "sentiment": sentiment,
            "feedback": feedback,
        }

        res = await client.post(
            f"https://{timbal_platform_config.host}/orgs/{org_id}/apps/{app_id}/runs/{run_id}/reactions", 
            headers=headers,
            json=payload,
        )
        res.raise_for_status()
