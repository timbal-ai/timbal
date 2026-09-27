"""Empty optional form objects must not prevent voice-server startup."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from timbal.server import voice as routes
from timbal.server.http import create_app
from timbal.voice.config import VoiceConfig, coerce_greeting, greeting_for_direction

FIELDS = ("greeting", "outbound_greeting", "user_idle", "ambient")


@pytest.mark.parametrize("field", FIELDS)
def test_empty_optional_block_is_unset_in_typed_config(field):
    raw = {"greeting": "Hello", field: {}}
    config = VoiceConfig(**raw)
    assert getattr(config, field) is None
    assert field not in config.model_fields_set
    assert raw[field] == {}  # Never mutate the caller's form state.
    if field == "outbound_greeting":
        assert greeting_for_direction(config, outbound=True).text == "Hello"


@pytest.mark.usefixtures("clear_voice_env")
@pytest.mark.parametrize("spelling", ["dict", "callable", "typed"])
def test_empty_blocks_preserve_server_defaults_and_declared_sparsity(monkeypatch, spelling):
    monkeypatch.setenv("TIMBAL_VOICE_GREETING", "Host greeting")
    monkeypatch.setenv("TIMBAL_VOICE_AMBIENT_SOURCE", "office")
    raw = {field: {} for field in FIELDS}
    declared = raw if spelling == "dict" else (lambda: raw) if spelling == "callable" else VoiceConfig(**raw)
    runnable = SimpleNamespace(voice_config=declared)
    merged = routes.merge_voice_config(runnable)
    assert merged.greeting.text == "Host greeting"
    assert merged.ambient.source == "office"
    assert merged.user_idle is None
    assert greeting_for_direction(merged, outbound=True).text == "Host greeting"
    assert routes.declared_voice_config(runnable) == {}


@pytest.mark.parametrize("with_defaults", [False, True])
def test_empty_client_blocks_preserve_defaults_and_outbound_inheritance(with_defaults):
    base = VoiceConfig(
        greeting="Hello" if with_defaults else None,
        user_idle={"text": "Still there?"} if with_defaults else None,
    )
    result = routes.merge_client_voice_overrides(base, {field: {} for field in FIELDS})
    assert result == base
    assert result.model_fields_set == base.model_fields_set
    assert "outbound_greeting" not in result.model_fields_set
    assert greeting_for_direction(result, outbound=True) == base.greeting


def test_empty_agent_blocks_preserve_all_inherited_behaviors(monkeypatch):
    base = VoiceConfig(
        greeting="Hello",
        outbound_greeting="Calling about your appointment",
        user_idle={"text": "Still there?"},
        ambient={"source": "office"},
    )
    monkeypatch.setattr(routes, "default_voice_config_from_env", lambda: base)
    merged = routes.merge_voice_config(SimpleNamespace(voice_config={field: {} for field in FIELDS}))
    assert merged.greeting == base.greeting
    assert merged.user_idle == base.user_idle
    assert merged.ambient == base.ambient
    # The env loader does not supply outbound_greeting. Its existing merge policy
    # excludes that field, so verify outbound inheritance through the client path.
    client_merged = routes.merge_client_voice_overrides(base, {field: {} for field in FIELDS})
    assert greeting_for_direction(client_merged, outbound=True) == base.outbound_greeting


def test_explicit_disable_and_documented_empty_defaults_still_work():
    config = VoiceConfig(greeting="Hello", outbound_greeting="", filler={}, recording={})
    assert greeting_for_direction(config, outbound=True) is None
    assert "outbound_greeting" in config.model_fields_set
    assert config.filler.enabled is True
    assert config.recording.layout == "mixed"
    assert coerce_greeting({}) is None
    assert coerce_greeting({"text": ""}) is None
    base = VoiceConfig(user_idle={"text": "Still there?"})
    assert routes.merge_client_voice_overrides(base, {"user_idle": ""}).user_idle is None


@pytest.mark.parametrize(
    "field,block",
    [
        ("user_idle", {"max_count": 0}),
        ("user_idle", {"text": "Still there?", "timeout_secs": -1}),
        ("greeting", {"delay_ms": 500}),
        ("outbound_greeting", {"delay_ms": 500}),
        ("ambient", {"volume": 0.2}),
        ("filler", {"delay_secs": -1}),
        ("user_idle", {"typo": True}),
    ],
)
def test_nonempty_invalid_blocks_still_fail_startup(field, block):
    with pytest.raises(ValidationError):
        routes.merge_voice_config(SimpleNamespace(voice_config={field: block}))


@pytest.mark.usefixtures("clear_voice_env")
def test_server_boots_and_serves_voice_config_with_empty_optional_blocks(monkeypatch, tmp_path):
    mod = tmp_path / "empty_voice_blocks_agent.py"
    mod.write_text(
        "from timbal import Agent\n"
        'agent = Agent(name="empty_blocks", model="timbal/TestModel", voice_config={\n'
        '    "language": "es", "greeting": {}, "outbound_greeting": {},\n'
        '    "user_idle": {}, "ambient": {}, "filler": {"enabled": True}\n'
        "})\n"
    )
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod}::agent")
    monkeypatch.setenv("TIMBAL_VOICE_WARMUP", "0")
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/healthcheck").status_code == 204
        declared = client.get("/voice_config").json()["voice_config"]
        assert declared == {"language": "es", "filler": {"enabled": True}}
        assert app.state.voice_config.user_idle is None
