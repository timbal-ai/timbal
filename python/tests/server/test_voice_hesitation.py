"""``VoiceConfig.hesitation`` from an agent's config to the session."""

from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import build_voice_session, declared_voice_config, merge_voice_config
from timbal.voice import HesitationConfig, VoiceConfig
from timbal.voice.config import DEFAULT_HESITATION_RATE


def make_agent(voice_config=None) -> Agent:
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    if voice_config is not None:
        agent.voice_config = {"turn_detector": "heuristic", **voice_config}
    return agent


def test_off_by_default():
    session, _ = build_voice_session(make_agent(), VoiceConfig(turn_detector="heuristic"), {})
    assert session.hesitation is None


def test_reaches_the_session_from_agent_config():
    agent = make_agent({"hesitation": {"rate": 0.3, "openers": {"es-ES": ["eh...", "mm,"]}, "seed": 5}})
    session, _ = build_voice_session(agent, merge_voice_config(agent), {})
    assert session.hesitation == HesitationConfig(rate=0.3, openers={"es": ["eh...", "mm,"]}, seed=5)
    assert session._hesitator.rate == 0.3
    assert session._hesitator.openers("es") == ("eh...", "mm,")


def test_true_is_the_default_rate():
    agent = make_agent({"hesitation": True})
    session, _ = build_voice_session(agent, merge_voice_config(agent), {})
    assert session.hesitation.rate == DEFAULT_HESITATION_RATE
    assert session.hesitation.openers is None


@pytest.mark.parametrize("block", [False, {}, {"rate": 0}, HesitationConfig()])
def test_an_off_block_builds_a_session_without_it(block):
    agent = make_agent({"hesitation": block})
    session, _ = build_voice_session(agent, merge_voice_config(agent), {})
    assert session.hesitation is None
    assert session._hesitator is None


def test_a_bad_block_fails_when_the_config_is_merged():
    with pytest.raises(ValueError):
        merge_voice_config(make_agent({"hesitation": {"openers": ["eh"]}}))


@pytest.mark.parametrize("hello", [True, {"rate": 1.0}, False, {}, {"rate": 0}])
def test_a_client_hello_cannot_set_or_clear_it(hello):
    off, _ = build_voice_session(make_agent(), VoiceConfig(turn_detector="heuristic"), {"hesitation": hello})
    assert off.hesitation is None
    defaults = VoiceConfig(turn_detector="heuristic", hesitation={"rate": 0.4})
    kept, _ = build_voice_session(make_agent(), defaults, {"hesitation": hello})
    assert kept.hesitation.rate == 0.4


@pytest.mark.parametrize(
    "declared,wire",
    [
        ({"hesitation": True}, {"hesitation": True}),
        ({"hesitation": {"rate": 0.3}}, {"hesitation": {"rate": 0.3}}),
        (
            {"hesitation": HesitationConfig(rate=0.3, openers=["hm,"])},
            {"hesitation": {"rate": 0.3, "openers": ["hm,"]}},
        ),
        (VoiceConfig(hesitation={"rate": 0.3}), {"hesitation": {"rate": 0.3}}),
        (VoiceConfig(hesitation=True), {"hesitation": {"rate": DEFAULT_HESITATION_RATE}}),
        ({"hesitation": {}}, {}),
        ({"hesitation": HesitationConfig()}, {}),
    ],
)
def test_the_declared_block_travels_to_a_remote_media_host(declared, wire):
    assert declared_voice_config(SimpleNamespace(voice_config=declared)) == wire
