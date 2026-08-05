import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.platform.integrations import Integration
from timbal.tools._creds import resolve_api_key
from timbal.tools.munsit import (
    MunsitListVoices,
    MunsitTextToSpeech,
)


@pytest.mark.asyncio
async def test_resolve_api_key_explicit():
    key = await resolve_api_key(
        env_var="MUNSIT_API_KEY",
        provider_name="Munsit",
        integration=None,
        api_key=SecretStr("local-key"),
    )
    assert key == "local-key"


@pytest.mark.asyncio
async def test_resolve_api_key_from_integration():
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"api_key": "platform-key"})
    key = await resolve_api_key(
        env_var="MUNSIT_API_KEY",
        provider_name="Munsit",
        integration=integration,
        api_key=None,
    )
    assert key == "platform-key"


@pytest.mark.asyncio
async def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MUNSIT_API_KEY", "env-key")
    key = await resolve_api_key(
        env_var="MUNSIT_API_KEY",
        provider_name="Munsit",
        integration=None,
        api_key=None,
    )
    assert key == "env-key"


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_text_to_speech_builds_request_and_encodes_wav():
    wav_bytes = b"RIFF....WAVEfmt "
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = wav_bytes
    mock_response.headers = {"content-type": "audio/wav"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = MunsitTextToSpeech(api_key=SecretStr("munsit-key"))
        out = await tool.handler(
            text="مرحبا بك في فصيح",
            voice_id="ar-najdi-male-2",
            model_id="faseeh-v1-preview",
            stability=0.5,
            speed=1.0,
            sample_rate=48000,
            dialect="auto",
        )

    assert out == {
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
        "content_type": "audio/wav",
        "sample_rate": 48000,
    }

    mock_client.post.assert_awaited_once()
    call = mock_client.post.await_args
    assert call.args[0] == "https://api.munsit.com/api/v1/text-to-speech/faseeh-v1-preview"
    assert call.kwargs["headers"] == {"x-api-key": "munsit-key"}
    assert call.kwargs["json"] == {
        "voice_id": "ar-najdi-male-2",
        "text": "مرحبا بك في فصيح",
        "stability": 0.5,
        "speed": 1.0,
        "streaming": False,
        "sample_rate": 48000,
        "dialect": "auto",
    }


@pytest.mark.asyncio
async def test_list_voices_hits_voices_endpoint():
    voices = [{"voice_id": "ar-najdi-male-2", "name": "Fahad"}]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = voices

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = MunsitListVoices(api_key=SecretStr("munsit-key"))
        out = await tool.handler()

    assert out == voices
    mock_client.get.assert_awaited_once()
    call = mock_client.get.await_args
    assert call.args[0] == "https://api.munsit.com/api/v1/voices"
    assert call.kwargs["headers"] == {"x-api-key": "munsit-key"}


@pytest.mark.asyncio
async def test_base_url_override():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = []

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = MunsitListVoices(
            api_key=SecretStr("uae-key"),
            base_url="https://ae.api.faseeh.ai/api/v1",
        )
        await tool.handler()

    call = mock_client.get.await_args
    assert call.args[0] == "https://ae.api.faseeh.ai/api/v1/voices"
