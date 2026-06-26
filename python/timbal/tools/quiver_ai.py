from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_BASE_URL = "https://api.quiver.ai/v1"
_DEFAULT_MODEL = "arrow-1.1"


def _normalize_references(references: list[str | dict[str, str]] | None) -> list[dict[str, str]] | None:
    """Accept URL strings or {url}/{base64} dicts; pass through as the API expects."""
    if not references:
        return None
    out: list[dict[str, str]] = []
    for ref in references:
        if isinstance(ref, str):
            out.append({"url": ref})
        elif isinstance(ref, dict):
            out.append(ref)
        else:
            raise ValueError(f"Invalid reference type: {type(ref).__name__}. Expected str (url) or dict.")
    return out


def _normalize_image(image: str | dict[str, str]) -> dict[str, str]:
    """Accept a URL string, base64 string, or dict with {url} or {base64}."""
    if isinstance(image, dict):
        return image
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            return {"url": image}
        return {"base64": image}
    raise ValueError(f"Invalid image type: {type(image).__name__}. Expected str or dict.")


class QuiverAIGenerateSVG(Tool):
    name: str = "quiver_ai_generate_svg"
    description: str | None = (
        "Generate one or more SVG graphics from a text prompt using QuiverAI's Arrow models. "
        "Supports optional reference images to guide style/composition. "
        "Returns raw SVG markup along with request metadata and credit usage."
    )
    integration: Annotated[str, Integration("quiver_ai")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _quiver_ai_generate_svg(
            prompt: str = Field(..., description="Primary text prompt that describes the desired SVG."),
            model: str = Field(_DEFAULT_MODEL, description='Model id, e.g. "arrow-1.1" (default), "arrow-1.1-max", "arrow-1".'),
            instructions: str | None = Field(None, description="Additional style/formatting guidance."),
            n: int = Field(1, description="Number of outputs to generate (1-16)."),
            references: list[str | dict[str, str]] | None = Field(
                None,
                description=(
                    "Optional reference images to guide style/composition. "
                    'Items can be a URL string, or {"url": "..."}, or {"base64": "..."}. '
                    "Max 4 for arrow-1.1, 16 for arrow-1.1-max."
                ),
            ),
            temperature: float | None = Field(None, description="Sampling temperature (0-2)."),
            top_p: float | None = Field(None, description="Nucleus sampling probability (0-1)."),
            presence_penalty: float | None = Field(None, description="Penalty for tokens already present (-2 to 2)."),
            max_output_tokens: int | None = Field(None, description="Upper bound for output token count (1-131072)."),
        ) -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="QuiverAI", env_var="QUIVERAI_API_KEY")
            import httpx

            payload: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "stream": False,
            }
            if instructions is not None:
                payload["instructions"] = instructions
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty
            if max_output_tokens is not None:
                payload["max_output_tokens"] = max_output_tokens
            normalized_refs = _normalize_references(references)
            if normalized_refs:
                payload["references"] = normalized_refs

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/svgs/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(120.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_quiver_ai_generate_svg, **kwargs)


class QuiverAIVectorizeSVG(Tool):
    name: str = "quiver_ai_vectorize_svg"
    description: str | None = (
        "Convert a raster image into an SVG using QuiverAI's Arrow models. "
        "Accepts an image URL or base64-encoded payload. "
        "Returns raw SVG markup along with request metadata and credit usage."
    )
    integration: Annotated[str, Integration("quiver_ai")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _quiver_ai_vectorize_svg(
            image: str | dict[str, str] = Field(
                ...,
                description=(
                    "Image to vectorize. Either a URL string (http/https), a base64 string, "
                    'or a dict like {"url": "..."} or {"base64": "..."}.'
                ),
            ),
            model: str = Field(_DEFAULT_MODEL, description='Model id, e.g. "arrow-1.1" (default).'),
            auto_crop: bool = Field(False, description="Auto-crop image to the dominant subject before vectorization."),
            target_size: int | None = Field(None, description="Square resize target in pixels (128-4096)."),
            temperature: float | None = Field(None, description="Sampling temperature (0-2)."),
            top_p: float | None = Field(None, description="Nucleus sampling probability (0-1)."),
            presence_penalty: float | None = Field(None, description="Penalty for tokens already present (-2 to 2)."),
            max_output_tokens: int | None = Field(None, description="Upper bound for output token count (1-131072)."),
        ) -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="QuiverAI", env_var="QUIVERAI_API_KEY")
            import httpx

            payload: dict[str, Any] = {
                "model": model,
                "image": _normalize_image(image),
                "stream": False,
                "auto_crop": auto_crop,
            }
            if target_size is not None:
                payload["target_size"] = target_size
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty
            if max_output_tokens is not None:
                payload["max_output_tokens"] = max_output_tokens

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/svgs/vectorizations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(120.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_quiver_ai_vectorize_svg, **kwargs)


class QuiverAIListModels(Tool):
    name: str = "quiver_ai_list_models"
    description: str | None = (
        "List all QuiverAI models available to the authenticated organization, "
        "including their ids, modalities, supported operations, and credit pricing."
    )
    integration: Annotated[str, Integration("quiver_ai")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _quiver_ai_list_models() -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="QuiverAI", env_var="QUIVERAI_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_quiver_ai_list_models, **kwargs)


class QuiverAIGetModel(Tool):
    name: str = "quiver_ai_get_model"
    description: str | None = "Retrieve metadata for a single QuiverAI model by id."
    integration: Annotated[str, Integration("quiver_ai")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _quiver_ai_get_model(
            model: str = Field(..., description='Model id, e.g. "arrow-1.1".'),
        ) -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="QuiverAI", env_var="QUIVERAI_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_BASE_URL}/models/{model}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=httpx.Timeout(30.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_quiver_ai_get_model, **kwargs)
