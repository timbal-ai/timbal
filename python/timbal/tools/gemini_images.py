import base64
from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_DEFAULT_EDIT_MODEL = "gemini-3.1-flash-image-preview"
_DEFAULT_GENERATE_MODEL = "gemini-3.1-flash-image-preview"
_DEFAULT_ANALYZE_MODEL = "gemini-3.1-flash-image-preview"


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


class EditImage(Tool):
    name: str = "gemini_edit_image"
    description: str | None = (
        "Edit or transform an existing image using a natural language prompt via Gemini. "
        "Returns the edited image as a base64 string."
    )
    integration: Annotated[str, Integration("gemini")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _edit_image(
            prompt: str,
            image_base64: str,
            image_mime_type: str = "image/png",
            model: str = _DEFAULT_EDIT_MODEL,
        ) -> Any:
            """
            prompt: instruction describing the desired edit, e.g. "remove the background".
            image_base64: the source image encoded as a base64 string.
            image_mime_type: MIME type of the input image, e.g. "image/png", "image/jpeg".
            model: Gemini model to use. Defaults to gemini-2.0-flash-exp-image-generation.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return _parse_gemini_response(response.json())

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GeminiImages/EditImage"
        super().__init__(handler=_edit_image, metadata=metadata, **kwargs)


class GenerateImage(Tool):
    name: str = "gemini_generate_image"
    description: str | None = (
        "Generate a new image from a text prompt using Gemini. "
        "Returns the generated image as a base64 string."
    )
    integration: Annotated[str, Integration("gemini")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _generate_image(
            prompt: str,
            negative_prompt: str | None = None,
            model: str = _DEFAULT_GENERATE_MODEL,
        ) -> Any:
            """
            prompt: text description of the image to generate.
            negative_prompt: optional description of what to avoid in the image.
            model: Gemini model to use. Defaults to gemini-2.0-flash-exp-image-generation.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            text = prompt
            if negative_prompt:
                text = f"{prompt}\n\nAvoid: {negative_prompt}"

            payload = {
                "contents": [
                    {
                        "parts": [{"text": text}]
                    }
                ],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_GEMINI_BASE}/{model}:generateContent",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return _parse_gemini_response(response.json())

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GeminiImages/GenerateImage"
        super().__init__(handler=_generate_image, metadata=metadata, **kwargs)


class AnalyzeImage(Tool):
    name: str = "gemini_analyze_image"
    description: str | None = (
        "Analyze, describe, or extract information from an image using Gemini vision. "
        "Returns a text response based on your question or instruction."
    )
    integration: Annotated[str, Integration("gemini")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _analyze_image(
            prompt: str,
            image_base64: str,
            image_mime_type: str = "image/png",
            model: str = _DEFAULT_ANALYZE_MODEL,
        ) -> Any:
            """
            prompt: question or instruction about the image,
                    e.g. "Describe this image", "What text is visible?", "List all objects".
            image_base64: the image encoded as a base64 string.
            image_mime_type: MIME type of the image, e.g. "image/png", "image/jpeg".
            model: Gemini model to use. Defaults to gemini-2.0-flash.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"Authorization": f"Bearer {token}"},
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GeminiImages/AnalyzeImage"
        super().__init__(handler=_analyze_image, metadata=metadata, **kwargs)


class GenerateImageWithReference(Tool):
    name: str = "gemini_generate_image_with_reference"
    description: str | None = (
        "Generate a new image using a text prompt and one or more reference images for style or composition guidance."
    )
    integration: Annotated[str, Integration("gemini")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _generate_image_with_reference(
            prompt: str,
            reference_images_base64: list[str],
            reference_mime_types: list[str] | None = None,
            model: str = _DEFAULT_GENERATE_MODEL,
        ) -> Any:
            """
            prompt: text instruction describing the desired output image.
            reference_images_base64: list of reference images as base64 strings.
            reference_mime_types: MIME type per reference image. Defaults to "image/png" for all.
            model: Gemini model to use. Defaults to gemini-2.0-flash-exp-image-generation.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            resolved_mime = reference_mime_types or ["image/png"] * len(reference_images_base64)

            parts: list[dict[str, Any]] = [{"text": prompt}]
            for b64, mime in zip(reference_images_base64, resolved_mime):
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{_GEMINI_BASE}/{model}:generateContent",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return _parse_gemini_response(response.json())

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GeminiImages/GenerateImageWithReference"
        super().__init__(handler=_generate_image_with_reference, metadata=metadata, **kwargs)


class ImageToBase64(Tool):
    name: str = "gemini_image_url_to_base64"
    description: str | None = (
        "Fetch an image from a public URL and return it as a base64 string, "
        "ready to use as input to other Gemini image tools."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _image_url_to_base64(
            url: str,
        ) -> Any:
            """
            url: publicly accessible URL of the image to fetch.
            """
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                mime_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
                b64 = base64.b64encode(response.content).decode("utf-8")
                return {"image_base64": b64, "mime_type": mime_type}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "GeminiImages/ImageToBase64"
        super().__init__(handler=_image_url_to_base64, metadata=metadata, **kwargs)
