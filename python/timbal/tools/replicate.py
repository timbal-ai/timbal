import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.replicate.com/v1"


async def _resolve_api_token(tool: Any) -> str:
    """Resolve Replicate API token from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        token = credentials.get("api_token")
        if token:
            return str(token)
    if tool.api_token is not None:
        return tool.api_token.get_secret_value()
    env_token = os.getenv("REPLICATE_API_TOKEN")
    if env_token:
        return env_token
    raise ValueError(
        "Replicate API token not found. Set REPLICATE_API_TOKEN environment variable, "
        "pass api_token in config, or configure an integration."
    )


def _replicate_headers(api_token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


class ReplicateCreatePrediction(Tool):
    name: str = "replicate_create_prediction"
    description: str | None = (
        "Create a Replicate prediction for a model version and inputs. "
        "Use official model refs like owner/name or a full version id. "
        "Optionally wait up to N seconds for completion via prefer_wait_seconds."
    )
    integration: Annotated[str, Integration("replicate")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_token": self.api_token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _replicate_create_prediction(
            version: str = Field(
                ...,
                description='Model version: "owner/model", "owner/model:version_hash", or 64-char version id.',
            ),
            input: dict[str, Any] = Field(
                ...,
                description="Input object for the model (see the model's API tab on replicate.com).",
            ),
            prefer_wait_seconds: int | None = Field(
                None,
                description="If set (1-60), sends Prefer: wait=n so the HTTP call may block until done or timeout.",
            ),
        ) -> dict[str, Any]:
            api_token = await _resolve_api_token(self)
            import httpx

            headers = _replicate_headers(api_token)
            if prefer_wait_seconds is not None:
                w = max(1, min(60, prefer_wait_seconds))
                headers["Prefer"] = f"wait={w}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/predictions",
                    headers=headers,
                    json={"version": version, "input": input},
                    timeout=httpx.Timeout(120.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_replicate_create_prediction, **kwargs)


class ReplicateGetPrediction(Tool):
    name: str = "replicate_get_prediction"
    description: str | None = "Get a Replicate prediction by id (status, output, error, urls)."
    integration: Annotated[str, Integration("replicate")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_token": self.api_token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _replicate_get_prediction(
            prediction_id: str = Field(..., description="Prediction id from replicate_create_prediction (often a UUID)."),
        ) -> dict[str, Any]:
            api_token = await _resolve_api_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/predictions/{prediction_id}",
                    headers=_replicate_headers(api_token),
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_replicate_get_prediction, **kwargs)


class ReplicateCancelPrediction(Tool):
    name: str = "replicate_cancel_prediction"
    description: str | None = "Cancel a running or starting Replicate prediction."
    integration: Annotated[str, Integration("replicate")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_token": self.api_token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _replicate_cancel_prediction(
            prediction_id: str = Field(..., description="Prediction id to cancel."),
        ) -> dict[str, Any]:
            api_token = await _resolve_api_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/predictions/{prediction_id}/cancel",
                    headers=_replicate_headers(api_token),
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                try:
                    return response.json()
                except Exception:
                    return {"ok": True, "detail": response.text or "cancelled"}

        super().__init__(handler=_replicate_cancel_prediction, **kwargs)


class ReplicateSearchModels(Tool):
    name: str = "replicate_search_models"
    description: str | None = (
        "Search public Replicate models, collections, and docs (beta API). "
        "Returns matching models with metadata such as tags and relevance score."
    )
    integration: Annotated[str, Integration("replicate")] | None = None
    api_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_token": self.api_token}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _replicate_search_models(
            query: str = Field(..., description="Search query string."),
            limit: int = Field(20, description="Max model results (1-50)."),
        ) -> dict[str, Any]:
            api_token = await _resolve_api_token(self)
            import httpx

            lim = max(1, min(50, limit))
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/search",
                    headers=_replicate_headers(api_token),
                    params={"query": query, "limit": lim},
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_replicate_search_models, **kwargs)
