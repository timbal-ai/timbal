import base64
import json
import os

import httpx
import pytest
from pydantic import SecretStr
from timbal.tools import OpenAISpeechToText, OpenAITextToSpeech


@pytest.fixture
def http(monkeypatch):
    requests, responses = [], []

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    client = httpx.AsyncClient
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs))
    return requests, responses


async def test_tts_defaults_and_credentials(http):
    requests, responses = http
    responses.append(httpx.Response(200, content=b"audio", headers={"content-type": "audio/mpeg"}))
    result = await OpenAITextToSpeech(api_key=SecretStr("key"))(text="Hello").collect()
    assert not result.error
    assert base64.b64decode(result.output["audio_base64"]) == b"audio"
    assert result.output["content_type"] == "audio/mpeg"
    assert requests[0].headers["Authorization"] == "Bearer key"
    assert json.loads(requests[0].content) == {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",
        "input": "Hello",
        "response_format": "mp3",
        "speed": 1.0,
    }


async def test_tts_options_validation_and_http_error(http, monkeypatch):
    requests, responses = http
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    tool = OpenAITextToSpeech()
    invalid = await tool(text="Hello", model="tts-1", instructions="Whisper").collect()
    assert invalid.error and not requests
    responses.append(httpx.Response(200, content=b"pcm"))
    result = await tool(text="Hello", voice="alloy", response_format="pcm", speed=0.9, instructions="Whisper").collect()
    assert not result.error
    assert requests[0].headers["Authorization"] == "Bearer env-key"
    assert json.loads(requests[0].content)["instructions"] == "Whisper"
    responses.append(httpx.Response(429, json={"error": {"message": "rate limited"}}))
    result = await tool(text="Hello").collect()
    assert result.error


@pytest.mark.parametrize("source", ["file", "base64"])
async def test_stt_uploads_audio_and_options(http, tmp_path, source):
    requests, responses = http
    responses.append(httpx.Response(200, json={"text": "Hello world"}))
    if source == "file":
        path = tmp_path / "sample.wav"
        path.write_bytes(b"audio-content")
        kwargs = {"audio_file_path": str(path)}
    else:
        kwargs = {"audio_file_base64": base64.b64encode(b"audio-content").decode(), "filename": "sample.wav"}
    result = await OpenAISpeechToText(api_key=SecretStr("key"))(**kwargs, language="en", prompt="Timbal").collect()
    assert not result.error and result.output == {"text": "Hello world"}
    request = requests[0]
    assert request.url.path == "/v1/audio/transcriptions"
    assert request.headers["Authorization"] == "Bearer key"
    assert "multipart/form-data" in request.headers["content-type"]
    assert b"audio-content" in request.content and b'filename="sample.wav"' in request.content
    for field in (b"gpt-transcribe", b"Timbal", b"json", b"en"):
        assert field in request.content


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"audio_file_path": "a.wav", "audio_file_base64": "eA=="},
        {"audio_file_base64": "eA=="},
        {"audio_file_base64": "!!!", "filename": "a.wav"},
    ],
)
async def test_stt_rejects_ambiguous_or_invalid_sources(http, kwargs):
    requests, _ = http
    result = await OpenAISpeechToText(api_key=SecretStr("key"))(**kwargs).collect()
    assert result.error and not requests


def test_tools_exports_config_and_provider_metadata():
    import timbal.tools as tools

    for cls in (OpenAISpeechToText, OpenAITextToSpeech):
        assert cls.__name__ in tools.__all__
        assert getattr(tools, cls.__name__) is cls
        tool = cls(api_key=SecretStr("key"))
        config = tool.get_config()
        assert config["name"]["value"] == tool.name
        assert "api_key" in config and "integration" in config
        assert config["integration"]["anyOf"][0]["x-timbal-integration"]["provider"] == "openai"


@pytest.mark.integration
async def test_live_openai_audio_tools():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set OPENAI_API_KEY for live audio tools")
    audio = await OpenAITextToSpeech()(text="The blue bicycle is outside.", response_format="wav").collect()
    assert not audio.error
    assert base64.b64decode(audio.output["audio_base64"]).startswith(b"RIFF")
    transcript = await OpenAISpeechToText()(
        audio_file_base64=audio.output["audio_base64"], filename="speech.wav", language="en"
    ).collect()
    assert not transcript.error
    assert "bicycle" in transcript.output["text"].lower()


@pytest.mark.parametrize("model", ["gpt-transcribe", "whisper-1"])
async def test_language_field_matches_file_transcription_model(http, model):
    requests, responses = http
    responses.append(httpx.Response(200, json={"text": "Hello"}))
    result = await OpenAISpeechToText(api_key=SecretStr("key"))(
        model=model,
        audio_file_base64="eA==",
        filename="test.wav",
        language="en",
    ).collect()
    assert not result.error
    expected = b'name="languages[]"' if model == "gpt-transcribe" else b'name="language"'
    assert expected in requests[0].content


@pytest.mark.parametrize("model,voice", [("tts-1", "marin"), ("tts-1-hd", "verse"), ("gpt-4o-mini-tts-typo", "coral")])
async def test_invalid_speech_combinations_fail_before_request(http, model, voice):
    requests, _ = http
    result = await OpenAITextToSpeech(api_key=SecretStr("key"))(text="Hello", model=model, voice=voice).collect()
    assert result.error and not requests
