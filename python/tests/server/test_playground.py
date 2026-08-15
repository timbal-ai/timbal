"""Playground launcher: stdlib LiveKit JWT mint + creds resolution."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from timbal.server.playground import ChildServer, livekit_creds, mint_livekit_jwt

_LIVEKIT_ENV = (
    "TIMBAL_LIVEKIT_URL",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "TIMBAL_LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "TIMBAL_LIVEKIT_API_SECRET",
)


class _FakeProc:
    """Just enough of a Popen for ``ChildServer.spawn`` to keep a handle."""

    pid = 1
    stdout = iter(())

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return 0


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
        url, key, secret = livekit_creds(tmp_path, launch_dir=tmp_path)
        assert (url, key, secret) == ("wss://file.example", "filekey", "filesecret")

        monkeypatch.setenv("LIVEKIT_API_KEY", "envkey")
        monkeypatch.setenv("TIMBAL_LIVEKIT_URL", "wss://lk.timbal.ai")
        url, key, secret = livekit_creds(tmp_path, launch_dir=tmp_path)
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
        assert livekit_creds(tmp_path, launch_dir=tmp_path) == ("wss://from-file", "k", "s")

    def test_launch_cwd_env_wins_over_agent_dir(self, tmp_path, monkeypatch) -> None:
        """Play is launched from a different tree than the agent file — the
        cwd ``.env`` is the one you actually edit."""
        for n in (
            "TIMBAL_LIVEKIT_URL",
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "TIMBAL_LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "TIMBAL_LIVEKIT_API_SECRET",
        ):
            monkeypatch.delenv(n, raising=False)
        agent_dir = tmp_path / "agent"
        launch_dir = tmp_path / "launch"
        agent_dir.mkdir()
        launch_dir.mkdir()
        (agent_dir / ".env").write_text(
            "LIVEKIT_URL=wss://agent.example\nLIVEKIT_API_KEY=agentk\nLIVEKIT_API_SECRET=agents\n",
            encoding="utf-8",
        )
        (launch_dir / ".env").write_text(
            "LIVEKIT_URL=wss://launch.example\nLIVEKIT_API_KEY=launchk\nLIVEKIT_API_SECRET=launchs\n",
            encoding="utf-8",
        )
        assert livekit_creds(agent_dir, launch_dir=launch_dir) == (
            "wss://launch.example",
            "launchk",
            "launchs",
        )

    def test_launch_cwd_env_fills_in_when_agent_has_none(self, tmp_path, monkeypatch) -> None:
        for n in (
            "TIMBAL_LIVEKIT_URL",
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "TIMBAL_LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "TIMBAL_LIVEKIT_API_SECRET",
        ):
            monkeypatch.delenv(n, raising=False)
        agent_dir = tmp_path / "agent"
        launch_dir = tmp_path / "launch"
        agent_dir.mkdir()
        launch_dir.mkdir()
        (launch_dir / ".env").write_text(
            "TIMBAL_LIVEKIT_URL=wss://lk.timbal.ai\nLIVEKIT_API_KEY=k\nLIVEKIT_API_SECRET=s\n",
            encoding="utf-8",
        )
        assert livekit_creds(agent_dir, launch_dir=launch_dir) == (
            "wss://lk.timbal.ai",
            "k",
            "s",
        )


class TestSpawnLoadsLaunchDotenv:
    """Play folds the launch-cwd ``.env`` into the child — that's the path
    that was dropping LIVEKIT_* when the agent file lived in ``examples/``."""

    def _agent(self, tmp_path):
        agent = tmp_path / "agent" / "demo.py"
        agent.parent.mkdir()
        agent.write_text("agent = None\n", encoding="utf-8")
        return agent

    def _spawn(self, monkeypatch, agent, *, transport: str = "ws") -> dict:
        captured: dict = {}

        def fake_popen(cmd, cwd=None, env=None, **kwargs):  # noqa: ARG001
            captured["env"] = env
            captured["cwd"] = cwd
            captured["cmd"] = cmd
            return _FakeProc()

        monkeypatch.setattr("timbal.server.playground.shutil.which", lambda _: "/usr/bin/uv")
        monkeypatch.setattr("timbal.server.playground.subprocess.Popen", fake_popen)
        child = ChildServer()
        child.spawn(f"{agent}::agent", port=59999, transport=transport)
        child.stop()
        return captured

    def test_cwd_env_reaches_the_child(self, tmp_path, monkeypatch) -> None:
        for n in _LIVEKIT_ENV:
            monkeypatch.delenv(n, raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        launch = tmp_path / "launch"
        launch.mkdir()
        (launch / ".env").write_text("ELEVENLABS_API_KEY=from-cwd\n", encoding="utf-8")
        monkeypatch.chdir(launch)
        captured = self._spawn(monkeypatch, self._agent(tmp_path))
        assert captured["env"]["ELEVENLABS_API_KEY"] == "from-cwd"

    def test_process_env_wins_over_cwd_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "from-process")
        launch = tmp_path / "launch"
        launch.mkdir()
        (launch / ".env").write_text("ELEVENLABS_API_KEY=from-cwd\n", encoding="utf-8")
        monkeypatch.chdir(launch)
        captured = self._spawn(monkeypatch, self._agent(tmp_path))
        assert captured["env"]["ELEVENLABS_API_KEY"] == "from-process"

    def test_livekit_spawn_reads_launch_cwd_env(self, tmp_path, monkeypatch) -> None:
        """The exact failure: agent in ``examples/``, keys in the repo-root ``.env``."""
        for n in _LIVEKIT_ENV:
            monkeypatch.delenv(n, raising=False)
        launch = tmp_path / "launch"
        launch.mkdir()
        (launch / ".env").write_text(
            "TIMBAL_LIVEKIT_URL=wss://lk.timbal.ai\nLIVEKIT_API_KEY=k\nLIVEKIT_API_SECRET=s\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(launch)
        captured = self._spawn(monkeypatch, self._agent(tmp_path), transport="livekit")
        assert captured["env"]["TIMBAL_LIVEKIT_URL"] == "wss://lk.timbal.ai"
        assert captured["env"]["TIMBAL_LIVEKIT_TOKEN"]
        assert captured["env"]["TIMBAL_VOICE_TRANSPORT"] == "livekit"
