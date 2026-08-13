"""Playground launcher: stdlib LiveKit JWT mint + creds resolution."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from timbal.server.playground import livekit_creds, mint_livekit_jwt


def _b64url_decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _verify(token: str, secret: str) -> tuple[dict, dict]:
    header_b64, payload_b64, sig_b64 = token.split(".")
    expected = hmac.new(
        secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
    ).digest()
    assert _b64url_decode(sig_b64) == expected
    header = json.loads(_b64url_decode(header_b64))
    payload = json.loads(_b64url_decode(payload_b64))
    return header, payload


class TestMintLivekitJwt:
    def test_hs256_roundtrip_and_grants(self) -> None:
        token = mint_livekit_jwt(
            api_key="APIxxx",
            api_secret="supersecret",
            identity="playground",
            room="pg-abc",
        )
        header, payload = _verify(token, "supersecret")
        assert header == {"alg": "HS256", "typ": "JWT"}
        assert payload["iss"] == "APIxxx"
        assert payload["sub"] == "playground"
        assert payload["video"] == {
            "roomJoin": True,
            "room": "pg-abc",
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        }
        assert payload["exp"] > payload["nbf"]
        assert payload["iat"] == payload["nbf"]


class TestLivekitCreds:
    def test_dotenv_then_env_override(self, tmp_path, monkeypatch) -> None:
        (tmp_path / ".env").write_text(
            'LIVEKIT_URL="wss://file.example"\n'
            "LIVEKIT_API_KEY=filekey\n"
            "LIVEKIT_API_SECRET=filesecret\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("TIMBAL_LIVEKIT_URL", raising=False)
        monkeypatch.delenv("LIVEKIT_URL", raising=False)
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("TIMBAL_LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        monkeypatch.delenv("TIMBAL_LIVEKIT_API_SECRET", raising=False)
        url, key, secret = livekit_creds(tmp_path)
        assert (url, key, secret) == ("wss://file.example", "filekey", "filesecret")

        monkeypatch.setenv("LIVEKIT_API_KEY", "envkey")
        monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "wss://lk.timbal.ai")
        url, key, secret = livekit_creds(tmp_path)
        assert url == "wss://lk.timbal.ai"
        assert key == "envkey"
        assert secret == "filesecret"

    def test_timbal_aliases(self, tmp_path, monkeypatch) -> None:
        for n in (
            "TIMBAL_LIVEKIT_URL",
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "TIMBAL_LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "TIMBAL_LIVEKIT_API_SECRET",
        ):
            monkeypatch.delenv(n, raising=False)
        (tmp_path / ".env").write_text(
            "TIMBAL_LIVEKIT_URL=wss://from-file\n"
            "TIMBAL_LIVEKIT_API_KEY=k\n"
            "TIMBAL_LIVEKIT_API_SECRET=s\n",
            encoding="utf-8",
        )
        assert livekit_creds(tmp_path) == ("wss://from-file", "k", "s")
