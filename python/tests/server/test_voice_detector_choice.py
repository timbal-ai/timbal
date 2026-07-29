"""Turn detector selection for served voice sessions.

The rules live in :func:`timbal.server.voice.select_turn_detector_spec` rather
than in the websocket handler precisely so they can be pinned here: they decide
what every deployment that doesn't configure a detector actually runs.
"""

from timbal.server.voice import select_turn_detector_spec
from timbal.voice.turn_detection import HeuristicTurnDetector, LocalAudioTurnDetector


class _FakeLocal:
    """Stand-in for what ``resolve_turn_detector("local")`` returns."""

    def __init__(self, audio_eou: object | None) -> None:
        self.audio_eou = audio_eou


def _with_extra(monkeypatch, *, available: bool) -> None:
    """Fake the presence of timbal[voice] via the resolver's degradation signal."""
    import timbal.voice.turn_detection as td

    monkeypatch.setattr(
        td,
        "resolve_turn_detector",
        lambda _spec=None: _FakeLocal(object() if available else None),
    )


class TestFluxOwnsEndpointing:
    def test_nothing_chosen_gets_provider(self):
        assert select_turn_detector_spec(None, None, stt_is_flux=True) == "provider"

    def test_local_is_overridden(self):
        assert select_turn_detector_spec("local", None, stt_is_flux=True) == "provider"

    def test_lexical_is_overridden(self):
        assert select_turn_detector_spec("lexical", None, stt_is_flux=True) == "provider"

    def test_an_instance_is_overridden_too(self):
        # Documents current behaviour: Flux wins even over an explicit instance
        # from voice_config, which is why the override is logged.
        assert select_turn_detector_spec(LocalAudioTurnDetector(), None, stt_is_flux=True) == "provider"

    def test_explicit_heuristic_is_respected(self):
        assert select_turn_detector_spec("heuristic", None, stt_is_flux=True) == "heuristic"

    def test_explicit_raw_is_respected(self):
        assert select_turn_detector_spec("raw", None, stt_is_flux=True) == "raw"


class TestSilenceEndpointingDefault:
    """Nova / ElevenLabs / anything that commits on a silence timeout."""

    def test_defaults_to_local_with_the_extra(self, monkeypatch):
        _with_extra(monkeypatch, available=True)
        assert select_turn_detector_spec(None, None, stt_is_flux=False) == "local"

    def test_falls_back_to_lexical_without_the_extra(self, monkeypatch):
        # `local` without an audio EOU model returns the heuristic decision
        # verbatim, so it would not hold; `lexical` does, with no extra deps.
        _with_extra(monkeypatch, available=False)
        assert select_turn_detector_spec(None, None, stt_is_flux=False) == "lexical"

    def test_never_defaults_to_the_holdless_heuristic(self, monkeypatch):
        for available in (True, False):
            _with_extra(monkeypatch, available=available)
            assert select_turn_detector_spec(None, None, stt_is_flux=False) != "heuristic"

    def test_explicit_heuristic_is_still_honoured(self):
        assert select_turn_detector_spec("heuristic", None, stt_is_flux=False) == "heuristic"

    def test_an_instance_is_passed_through(self):
        detector = HeuristicTurnDetector()
        assert select_turn_detector_spec(detector, None, stt_is_flux=False) is detector

    def test_a_factory_is_passed_through(self):
        def factory() -> HeuristicTurnDetector:
            return HeuristicTurnDetector()

        assert select_turn_detector_spec(factory, None, stt_is_flux=False) is factory


class TestClientOverride:
    def test_client_mode_beats_the_server_spec(self):
        assert select_turn_detector_spec("local", "heuristic", stt_is_flux=False) == "heuristic"

    def test_client_mode_beats_a_server_instance(self):
        assert select_turn_detector_spec(LocalAudioTurnDetector(), "lexical", stt_is_flux=False) == "lexical"

    def test_blank_client_mode_is_ignored(self):
        assert select_turn_detector_spec("lexical", "   ", stt_is_flux=False) == "lexical"

    def test_non_string_client_spec_is_refused(self):
        # A browser must not be able to hand the server a callable.
        assert select_turn_detector_spec("lexical", {"mode": "local"}, stt_is_flux=False) == "lexical"

    def test_client_cannot_escape_the_flux_override(self):
        assert select_turn_detector_spec(None, "local", stt_is_flux=True) == "provider"
