import os
from typing import Annotated, Any
from urllib.parse import quote, urlparse

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_QUEUE_BASE = "https://queue.fal.run"


async def _resolve_fal_key(tool: Any) -> str:
    """Resolve fal API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        key = credentials.get("api_key")
        if key:
            return str(key)
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("FAL_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "fal API key not found. Set FAL_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


def _fal_auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}


def _encode_model_path(model_id: str) -> str:
    """Model id may contain slashes; path segments must be encoded for URLs."""
    return "/".join(quote(part, safe="") for part in model_id.split("/"))


def _queue_model_path_from_submit_response(data: dict[str, Any]) -> str | None:
    """fal may normalize the endpoint (e.g. flux/schnell → flux); URLs carry the canonical path."""
    url = data.get("status_url") or data.get("response_url")
    if not isinstance(url, str) or "/requests/" not in url:
        return None
    try:
        path = urlparse(url).path.strip("/").split("/")
        i = path.index("requests")
        if i > 0:
            return "/".join(path[:i])
    except ValueError:
        return None
    return None


class FalQueueSubmit(Tool):
    name: str = "fal_queue_submit"
    description: str | None = (
        "Submit an async inference job to fal's queue for a model (e.g. fal-ai/flux/schnell). "
        "Returns request_id, status_url, response_url, cancel_url, and queue_model_path (canonical path for "
        "follow-up calls). Important: fal may rewrite the route—e.g. you submit fal-ai/flux/schnell but URLs use "
        "fal-ai/flux. For fal_queue_status, fal_queue_result, and fal_queue_cancel always pass model_id="
        "queue_model_path from this response (or the path segment from status_url before /requests/), not the "
        "original submit string, or status polling can return HTTP 405."
    )
    integration: Annotated[str, Integration("fal")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _fal_queue_submit(
            model_id: str = Field(
                ...,
                description='fal model id, e.g. "fal-ai/flux/schnell" or "fal-ai/flux/dev"',
            ),
            input: dict[str, Any] = Field(
                ...,
                description="Model input as JSON (fields depend on the model; image models often use 'prompt').",
            ),
            path_suffix: str = Field(
                "",
                description='Optional sub-path appended to the model endpoint (e.g. "/custom-endpoint").',
            ),
            webhook_url: str | None = Field(
                None,
                description="If set, fal POSTs the result to this URL when the job completes.",
            ),
        ) -> dict[str, Any]:
            api_key = await _resolve_fal_key(self)
            import httpx

            path = _encode_model_path(model_id)
            if path_suffix:
                path = f"{path}{path_suffix if path_suffix.startswith('/') else '/' + path_suffix}"
            url = f"{_QUEUE_BASE}/{path}"
            if webhook_url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}fal_webhook={quote(webhook_url, safe='')}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=_fal_auth_headers(api_key),
                    json=input,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    qmp = _queue_model_path_from_submit_response(payload)
                    if qmp:
                        payload["queue_model_path"] = qmp
                return payload

        super().__init__(handler=_fal_queue_submit, **kwargs)


class FalQueueStatus(Tool):
    name: str = "fal_queue_status"
    description: str | None = (
        "Poll fal queue status for a submitted request (IN_QUEUE, IN_PROGRESS, COMPLETED, or error). "
        "model_id must be the canonical queue path from fal_queue_submit's queue_model_path (or parsed from "
        "status_url), not necessarily the same string you used to submit—wrong path causes HTTP 405."
    )
    integration: Annotated[str, Integration("fal")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _fal_queue_status(
            model_id: str = Field(
                ...,
                description=(
                    "Canonical queue model path: use queue_model_path from fal_queue_submit, or the path between "
                    "https://queue.fal.run/ and /requests/ in status_url. Often differs from the submit model_id."
                ),
            ),
            request_id: str = Field(..., description="request_id returned by fal_queue_submit."),
            include_logs: bool = Field(False, description="If True, append ?logs=1 for runner log lines."),
        ) -> dict[str, Any]:
            api_key = await _resolve_fal_key(self)
            import httpx

            path = _encode_model_path(model_id)
            url = f"{_QUEUE_BASE}/{path}/requests/{request_id}/status"
            if include_logs:
                url = f"{url}?logs=1"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Key {api_key}"},
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_fal_queue_status, **kwargs)


class FalQueueResult(Tool):
    name: str = "fal_queue_result"
    description: str | None = (
        "Fetch the completed output for a fal queue request once status is COMPLETED. Output shape is model-specific. "
        "Use the same model_id as for fal_queue_status (queue_model_path from submit, not necessarily the original "
        "submit model_id)."
    )
    integration: Annotated[str, Integration("fal")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _fal_queue_result(
            model_id: str = Field(
                ...,
                description=(
                    "Canonical queue model path from fal_queue_submit.queue_model_path or status_url (see "
                    "fal_queue_status)."
                ),
            ),
            request_id: str = Field(..., description="request_id returned by fal_queue_submit."),
        ) -> dict[str, Any]:
            api_key = await _resolve_fal_key(self)
            import httpx

            path = _encode_model_path(model_id)
            url = f"{_QUEUE_BASE}/{path}/requests/{request_id}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Key {api_key}"},
                    timeout=httpx.Timeout(120.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_fal_queue_result, **kwargs)


class FalQueueCancel(Tool):
    name: str = "fal_queue_cancel"
    description: str | None = (
        "Cancel a fal queue request (in queue or in progress). "
        "Use queue_model_path / same model_id as status and result, not necessarily the original submit id."
    )
    integration: Annotated[str, Integration("fal")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _fal_queue_cancel(
            model_id: str = Field(
                ...,
                description=(
                    "Canonical queue model path from fal_queue_submit.queue_model_path or status_url (see "
                    "fal_queue_status)."
                ),
            ),
            request_id: str = Field(..., description="request_id returned by fal_queue_submit."),
        ) -> dict[str, Any]:
            api_key = await _resolve_fal_key(self)
            import httpx

            path = _encode_model_path(model_id)
            url = f"{_QUEUE_BASE}/{path}/requests/{request_id}/cancel"

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    url,
                    headers={"Authorization": f"Key {api_key}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                try:
                    body: Any = response.json()
                except Exception:
                    body = {"detail": response.text}
                if not isinstance(body, dict):
                    body = {"data": body}
                if response.status_code in (202, 400, 404):
                    return {**body, "http_status": response.status_code}
                response.raise_for_status()
                return body

        super().__init__(handler=_fal_queue_cancel, **kwargs)
