from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Image generation/editing: stable GA model (listed as "Nano Banana 2" in the Gemini API).
# Preview sibling gemini-3.1-flash-image-preview still works but is not the stable default.
_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
# Vision analysis: general multimodal flash — no image output modality needed.
_DEFAULT_ANALYZE_MODEL = "gemini-3.5-flash"


def _gemini_headers(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def _parse_gemini_response(data: dict[str, Any]) -> dict[str, Any]:
    """Extract text and image parts from a Gemini generateContent response."""
    result: dict[str, Any] = {"text": None, "image_base64": None, "mime_type": None}
    candidates = data.get("candidates", [])
    if not candidates:
        return result
    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        if "text" in part and result["text"] is None:
            result["text"] = part["text"]
        if "inlineData" in part and result["image_base64"] is None:
            result["image_base64"] = part["inlineData"].get("data")
            result["mime_type"] = part["inlineData"].get("mimeType", "image/png")
    return result


class GeminiImagesEditImage(Tool):
    name: str = "gemini_edit_image"
    description: str | None = (
        "Edit or transform an existing image using a natural language prompt via Gemini. "
        "Returns the edited image as a base64 string."
    )
    integration: Annotated[str, Integration("gemini")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _edit_image(
            prompt: str = Field(
                ..., description="Instruction describing the desired edit, e.g. 'remove the background'."
            ),
            image_base64: str = Field(..., description="The source image encoded as a base64 string."),
            image_mime_type: str = Field(
                "image/png", description="MIME type of the input image, e.g. 'image/png', 'image/jpeg'."
            ),
            model: str = Field(
                _DEFAULT_IMAGE_MODEL,
                description=f"Gemini image model. Defaults to {_DEFAULT_IMAGE_MODEL}.",
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="Gemini", env_var="GEMINI_API_KEY")
            import httpx

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": image_mime_type,
                                    "data": image_base64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_GEMINI_BASE}/{model}:generateContent",
                    headers=_gemini_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return _parse_gemini_response(response.json())

        super().__init__(handler=_edit_image, **kwargs)


class GeminiImagesGenerateImage(Tool):
    name: str = "gemini_generate_image"
    description: str | None = (
        "Generate a new image from a text prompt using Gemini. Returns the generated image as a base64 string."
    )
    integration: Annotated[str, Integration("gemini")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _generate_image(
            prompt: str = Field(..., description="Text description of the image to generate."),
            negative_prompt: str | None = Field(
                None, description="Optional description of what to avoid in the image."
            ),
            model: str = Field(
                _DEFAULT_IMAGE_MODEL,
                description=f"Gemini image model. Defaults to {_DEFAULT_IMAGE_MODEL}.",
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="Gemini", env_var="GEMINI_API_KEY")
            import httpx

            text = prompt
            if negative_prompt:
                text = f"{prompt}\n\nAvoid: {negative_prompt}"

            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_GEMINI_BASE}/{model}:generateContent",
                    headers=_gemini_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return _parse_gemini_response(response.json())

        super().__init__(handler=_generate_image, **kwargs)


class GeminiImagesAnalyzeImage(Tool):
    name: str = "gemini_analyze_image"
    description: str | None = (
        "Analyze, describe, or extract information from an image using Gemini vision. "
        "Returns a text response based on your question or instruction."
    )
    integration: Annotated[str, Integration("gemini")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _analyze_image(
            prompt: str = Field(
                ...,
                description="Question or instruction about the image, e.g. 'Describe this image', 'What text is visible?', 'List all objects'.",
            ),
            image_base64: str = Field(..., description="The image encoded as a base64 string."),
            image_mime_type: str = Field(
                "image/png", description="MIME type of the image, e.g. 'image/png', 'image/jpeg'."
            ),
            model: str = Field(
                _DEFAULT_ANALYZE_MODEL,
                description=f"Gemini vision model. Defaults to {_DEFAULT_ANALYZE_MODEL}.",
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="Gemini", env_var="GEMINI_API_KEY")
            import httpx

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": image_mime_type,
                                    "data": image_base64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {"responseModalities": ["TEXT"]},
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{_GEMINI_BASE}/{model}:generateContent",
                    headers=_gemini_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = " ".join(p.get("text", "") for p in parts if "text" in p)
                return {"text": text}

        super().__init__(handler=_analyze_image, **kwargs)


class GeminiImagesGenerateImageWithReference(Tool):
    name: str = "gemini_generate_image_with_reference"
    description: str | None = (
        "Generate a new image using a text prompt and one or more reference images for style or composition guidance."
    )
    integration: Annotated[str, Integration("gemini")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _generate_image_with_reference(
            prompt: str = Field(..., description="Text instruction describing the desired output image."),
            reference_images_base64: list[str] = Field(..., description="List of reference images as base64 strings."),
            reference_mime_types: list[str] | None = Field(
                None, description="MIME type per reference image. Defaults to 'image/png' for all."
            ),
            model: str = Field(
                _DEFAULT_IMAGE_MODEL,
                description=f"Gemini image model. Defaults to {_DEFAULT_IMAGE_MODEL}.",
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="Gemini", env_var="GEMINI_API_KEY")
            import httpx

            resolved_mime = reference_mime_types or ["image/png"] * len(reference_images_base64)

            parts: list[dict[str, Any]] = [{"text": prompt}]
            for b64, mime in zip(reference_images_base64, resolved_mime, strict=True):
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_GEMINI_BASE}/{model}:generateContent",
                    headers=_gemini_headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
                return _parse_gemini_response(response.json())

        super().__init__(handler=_generate_image_with_reference, **kwargs)


class GeminiImagesImageToBase64(Tool):
    name: str = "gemini_image_url_to_base64"
    description: str | None = (
        "Fetch an image from a public URL and return it as a base64 string, "
        "ready to use as input to other Gemini image tools."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _image_url_to_base64(
            url: str = Field(..., description="Publicly accessible URL of the image to fetch."),
        ) -> Any:
            import base64

            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                mime_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
                b64 = base64.b64encode(response.content).decode("utf-8")
                return {"image_base64": b64, "mime_type": mime_type}

        super().__init__(handler=_image_url_to_base64, **kwargs)
