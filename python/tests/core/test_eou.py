"""Tests for end-of-utterance prediction (timbal.voice.eou)."""

from __future__ import annotations

from timbal.voice.eou import EouPredictor, PunctuationEouPredictor, TextEouPredictor


class TestTextEouAlias:
    def test_eou_predictor_alias(self) -> None:
        assert EouPredictor is TextEouPredictor


class TestPunctuationEouPredictor:
    async def test_empty_is_complete(self) -> None:
        p = PunctuationEouPredictor()
        assert await p.predict_eou("") == 1.0
        assert await p.predict_eou("   ") == 1.0

    async def test_terminal_punctuation_complete(self) -> None:
        p = PunctuationEouPredictor()
        for text in ("Tell me a story.", "How are you?", "Stop!", "Vale…", "¿Qué tal?"):
            assert await p.predict_eou(text) == p.P_TERMINAL

    async def test_continuing_punctuation_incomplete(self) -> None:
        p = PunctuationEouPredictor()
        for text in ("I want to say,", "First this;", "Listen —"):
            assert await p.predict_eou(text) == p.P_CONTINUING

    async def test_trailing_dangling_token_incomplete_english(self) -> None:
        p = PunctuationEouPredictor()
        for text in (
            "I was wondering if you could tell me about",
            "Can you help me with",
            "I think that",
            "Give it to the",
        ):
            assert await p.predict_eou(text) == p.P_DANGLING

    async def test_trailing_dangling_token_incomplete_spanish(self) -> None:
        p = PunctuationEouPredictor()
        for text in ("Quiero saber sobre el", "Me gustaría hablar de", "Puedes ayudarme con"):
            assert await p.predict_eou(text) == p.P_DANGLING

    async def test_content_word_no_punct_is_neutral(self) -> None:
        p = PunctuationEouPredictor()
        # A complete-looking clause STT dropped the period on.
        assert await p.predict_eou("Tell me a story") == p.P_NEUTRAL
        assert await p.predict_eou("Quiero un café") == p.P_NEUTRAL

    async def test_neutral_leans_complete_dangling_leans_incomplete(self) -> None:
        p = PunctuationEouPredictor()
        assert await p.predict_eou("Tell me a story") >= 0.5
        assert await p.predict_eou("I was wondering about the") < 0.5

    async def test_case_insensitive_dangling(self) -> None:
        p = PunctuationEouPredictor()
        assert await p.predict_eou("Tell me about THE") == p.P_DANGLING

    async def test_subclass_can_tune_scores(self) -> None:
        class Strict(PunctuationEouPredictor):
            P_NEUTRAL = 0.3

        assert await Strict().predict_eou("Tell me a story") == 0.3

    async def test_lifecycle_noops(self) -> None:
        p = PunctuationEouPredictor()
        await p.start()
        await p.close()
