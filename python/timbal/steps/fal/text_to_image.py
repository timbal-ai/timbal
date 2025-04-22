import asyncio

import os

import httpx

from ...errors import APIKeyNotFoundError
from ...types.field import Field
from ...types.file import File


async def gen_images(
    model: str = Field(
        # TODO Add enums.
        default="fal-ai/flux-pro/v1.1-ultra", # 0.06 x image
        description="Model to use for image generation.",
    ),
    prompt: str = Field(
        description="The prompt to generate an image",
    ),
    # TODO Add more parameters.
) -> list[File]:

    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        raise APIKeyNotFoundError("FAL_KEY not found")

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Key {fal_key}"}

        res = await client.post(
            f"https://queue.fal.run/{model}",
            headers=headers,
            json={
                "prompt": prompt,
                "raw": False,
            }
        )

        res.raise_for_status()
        res_json = res.json()

        response_url = res_json["response_url"]
        status_url = res_json["status_url"]

        images = []
        while not images:
            res = await client.get(status_url, headers=headers)

            res.raise_for_status()
            res_json = res.json()

            if res_json["status"] == "COMPLETED":
                res = await client.get(response_url, headers=headers)

                res.raise_for_status()
                res_json = res.json()
                # {'images': [{'url': 'https://fal.media/files/penguin/500-x3utjpEEQlxryzxIw_8b9db5661adb4ff48cfd858cd0bc177d.jpg', 'width': 2752, 'height': 1536, 'content_type': 'image/jpeg'}], 'timings': {}, 'seed': 2784808122, 'has_nsfw_concepts': [False], 'prompt': 'a green ferrari'}
                images = res_json["images"]
            
            await asyncio.sleep(0.1)
                
    # TODO Return the usage.
    return [File.validate(image["url"]) for image in images]
