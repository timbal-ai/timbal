import base64
import json
import os

import httpx
import pytest
from pydantic import SecretStr
from timbal.tools import DeepgramListVoices, DeepgramTextToSpeech


@pytest.fixture
def requests(monkeypatch):
    seen = []
    responses = []

    def handle(request):
        seen.append(request)
        return responses.pop(0)

    client = httpx.AsyncClient
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs))
    return seen, responses


async def test_tts_tool_generates_audio_with_framework_defaults(requests):
    seen, responses = requests
    responses.append(httpx.Response(200, content=b"audio", headers={"content-type": "audio/mpeg"}))
    tool = DeepgramTextToSpeech(api_key=SecretStr("key"))
    result = await tool(text="Hello").collect()
    assert not result.error
    assert base64.b64decode(result.output["audio_base64"]) == b"audio"
    assert result.output["content_type"] == "audio/mpeg"
    request = seen[0]
    assert request.url.path == "/v1/speak"
    assert request.headers["Authorization"] == "Token key"
    assert dict(request.url.params) == {"model": "aura-2-thalia-en", "encoding": "mp3", "speed": "1.0"}
    assert json.loads(request.content) == {"text": "Hello"}


async def test_tts_tool_optional_formats_env_and_http_errors(requests, monkeypatch):
    seen, responses = requests
    monkeypatch.setenv("DEEPGRAM_API_KEY", "env-key")
    responses.append(httpx.Response(200, content=b"pcm", headers={"content-type": "audio/l16"}))
    tool = DeepgramTextToSpeech()
    result = await tool(
        text="Hola", model="aura-2-celeste-es", encoding="linear16", sample_rate=16000, container="none", speed=0.9
    ).collect()
    assert not result.error
    assert seen[0].headers["Authorization"] == "Token env-key"
    assert seen[0].url.params["container"] == "none"
    assert seen[0].url.params["sample_rate"] == "16000"
    responses.append(httpx.Response(401, json={"err_msg": "Invalid credentials"}))
    result = await tool(text="Hello").collect()
    assert result.error


async def test_voice_discovery_preserves_metadata_and_filters_language(requests):
    seen, responses = requests
    spanish = {"canonical_name": "aura-2-celeste-es", "languages": ["es-MX"], "metadata": {"sample": "sample.wav"}}
    english = {"canonical_name": "aura-2-thalia-en", "languages": ["en-US"]}
    responses.append(httpx.Response(200, json={"stt": [{"name": "nova-3"}], "tts": [spanish, english]}))
    result = await DeepgramListVoices(api_key=SecretStr("key"))(language="ES").collect()
    assert not result.error
    assert result.output == {"voices": [spanish]}
    assert seen[0].url.path == "/v1/models"


def test_tools_are_discoverable_and_configurable():
    import timbal.tools as tools

    for cls in (DeepgramTextToSpeech, DeepgramListVoices):
        assert cls.__name__ in tools.__all__
        assert getattr(tools, cls.__name__) is cls
        tool = cls(api_key=SecretStr("key"))
        assert tool.get_config()["name"]["value"] == tool.name


@pytest.mark.integration
async def test_live_voice_discovery_and_audio_generation():
    if not os.getenv("DEEPGRAM_API_KEY"):
        pytest.skip("Set DEEPGRAM_API_KEY to run live Aura tests")
    voices = await DeepgramListVoices()(language="en").collect()
    assert not voices.error
    assert any(v["canonical_name"] == "aura-2-thalia-en" for v in voices.output["voices"])
    audio = await DeepgramTextToSpeech()(text="Hello from Timbal.").collect()
    assert not audio.error
    assert audio.output["content_type"].startswith("audio/")
    assert len(base64.b64decode(audio.output["audio_base64"])) > 1000
