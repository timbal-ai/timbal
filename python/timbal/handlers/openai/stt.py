import os
from typing import Any

from openai import AsyncOpenAI

from ...errors import APIKeyNotFoundError
from ...types.field import Field
from ...types.file import File


async def stt(
    audio_file: File = Field(
        description=(
            "The audio file to transcribe. "
            "All major audio and video formats are supported. "
            "The file size must be less than 1GB."
        ),
    ),
    model_id: str = Field(
        default="gpt-4o-mini-transcribe",
        description="The OpenAI model to use for the STT.",
        choices=["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
    ),
    chunking_strategy: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Controls how the audio is cut into chunks."
            "- set to 'auto' automatically set chunking parameters based on the audio."
            "- set to an object if you want to tweak VAD detection parameters manually: "
            "    type: 'server_vad'"
            "    prefix_padding_ms: integer (Optional) -Amount of audio to include before the VAD detected speech (in milliseconds). Defaults to 300."
            "    silence_duration_ms: integer (Optional) -Amount of silence to detect speech stop (in milliseconds). With shorter values the model will respond more quickly, but may jump in on short pauses from the user. Defaults to 200."
            "    threshold: number (Optional) -Sensitivity threshold (0.0 to 1.0) for voice activity detection. A higher threshold will require louder audio to activate the model, and thus might perform better in noisy environments. Defaults to 0.5."
        ),
    ),
    include: list[str] | None = Field(
        default=None,
        description="A list of additional information to include in the response. e.g. logprobs -  only works with response_format set to json, and models gpt-4o-transcribe and gpt-4o-mini-transcribe.",
    ),
    language: str | None = Field(
        default=None,
        description="The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.",
    ),
    prompt: str | None = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.",
    ),
    response_format: str = Field(
        default="json",
        description="The format of the response. For gpt-4o-transcribe and gpt-4o-mini-transcribe, the only supported format is json.",
        choices=["json", "text", "srt", "verbose_json", "vtt"],
    ),
    stream: bool | None= Field(
        default=False,
        description=(
            "Whether to stream the response or not."
            "Streaming is not supported for the whisper-1 model and will be ignored.",
        ),
    ),
    temperature: float | None = Field(
        default=0,
        description=(
            "The sampling temperature between 0 and 1. Higher values make the output more random."
            "If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit."
        ),
    ),
    timestamp_granularities: list[str] | None = Field(
        default=["segment"],
        description=(
            "The timestamp granularities to populate for this transcription. 'response_format' must be set to 'verbose_json'."
            "Either or both of these options are supported: word, or segment"
        ),
    ),
) -> str: # ? Should we return the other elements.

    # Enable calling this step without pydantic model_validate()
    audio_file = audio_file.default if hasattr(audio_file, "default") else audio_file
    model_id = model_id.default if hasattr(model_id, "default") else model_id
    chunking_strategy = chunking_strategy.default if hasattr(chunking_strategy, "default") else chunking_strategy
    include = include.default if hasattr(include, "default") else include
    language = language.default if hasattr(language, "default") else language
    prompt = prompt.default if hasattr(prompt, "default") else prompt
    response_format = response_format.default if hasattr(response_format, "default") else response_format
    stream = stream.default if hasattr(stream, "default") else stream
    temperature = temperature.default if hasattr(temperature, "default") else temperature
    timestamp_granularities = timestamp_granularities.default if hasattr(timestamp_granularities, "default") else timestamp_granularities

    args = {}
    if chunking_strategy:
        args["chunking_strategy"] = chunking_strategy
    if include:
        args["include"] = include
    if language:
        args["language"] = language
    if prompt:
        args["prompt"] = prompt

    if not audio_file.__content_type__.startswith("audio"):
        raise ValueError(f"STT expected an audio file content type. Got {audio_file.__content_type__}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("OPENAI_API_KEY not found")
    
    client = AsyncOpenAI(api_key=api_key)
    
    transcript = await client.audio.transcriptions.create(
        model=model_id,
        file=audio_file,
        stream=stream,
        temperature=temperature,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
    )

    return transcript.text
