"""Sentence and clause boundaries for the first TTS segment.

Rules ported from LiveKit Agents ``tokenize/_basic_sent.py`` (Apache-2.0) and
the Pipecat ``match_endofsentence`` cases (BSD-2-Clause): abbreviations,
initials, decimals, and thousands separators are not cuts. The first segment
may still close on a comma after four words. Later segments stay on sentences.
"""

from timbal.voice.segments import clause_boundary_ends, sentence_boundary_ends
from timbal.voice.session import FIRST_SEGMENT_MIN_WORDS, _flush_segment


def _only_sentence(text: str) -> str:
    ends = sentence_boundary_ends(text)
    assert len(ends) == 1, ends
    return text[: ends[0]]


class TestSentenceBoundaries:
    """Pipecat's end-of-sentence cases, plus Spanish titles. A comma is never a sentence."""

    def test_plain_sentence(self) -> None:
        assert _only_sentence("This is a sentence. This is another one").rstrip() == "This is a sentence."
        assert _only_sentence("This is a sentence! Next").rstrip().endswith("!")
        assert _only_sentence("This is a sentence? Next").rstrip().endswith("?")

    def test_titles_and_acronyms_are_not_cuts(self) -> None:
        samples = [
            "This is for Mr. and Mrs. Jones. Next",
            "Ask Dr. Smith and Sra. García about it. Next",
            "Prof. Walker will see you today. Next",
            "U.S.A. and N.A.S.A. launched. Next",
            "EE. UU. firmaron el acuerdo ayer. Luego",
            "The number pi is 3.14159. Next",
            "The host is 192.168.1.1 today. Next",
            "My number is 123-456-7890. Next",
            "It is still early, it's 3:00 a.m. today. Next",
            "Son las 3:00 p.m. en Madrid hoy. Luego",
            "Acme Inc. ships tomorrow morning. Thanks",
            "El total es 1.234,56 euros exactos. Luego",
        ]
        for text in samples:
            cut = _only_sentence(text)
            assert "Next" not in cut and "Luego" not in cut and "Thanks" not in cut
            assert not cut.rstrip().endswith(("Mr.", "Dr.", "Sra.", "Prof.", "a.m.", "p.m.", "Inc."))

    def test_no_sentence_yet(self) -> None:
        for text in (
            "This is not a sentence",
            "This is not a sentence,",
            "Ok, Mr. Smith let's ",
            "Dr. Walker, I presume ",
            "America, or the U.S",
            "The number pi is 3.14159",
            "Heute ist Dienstag, der 3",
            "Ask Dr. Smith about the results today",
        ):
            assert sentence_boundary_ends(text) == []

    def test_number_label_is_not_a_sentence(self) -> None:
        assert _only_sentence("See No. 5 in the list today. Next").rstrip().endswith("today.")
        text = "No. Thanks for waiting on this. Next"
        first = text[: sentence_boundary_ends(text)[0]]
        assert first.rstrip() == "No."

    def test_contraction_period_is_a_sentence(self) -> None:
        assert _only_sentence("I don't. Next").rstrip() == "I don't."

    def test_comma_is_not_a_sentence_boundary(self) -> None:
        text = "The return window is thirty days, so you can still send it back. Next"
        assert _only_sentence(text).rstrip().endswith("back.")
        assert clause_boundary_ends(text)
        assert text[: clause_boundary_ends(text)[0]].rstrip().endswith("days,")


class TestFirstSegmentComma:
    def test_long_clause_flushes_on_the_comma(self) -> None:
        text = "The return window is thirty days, so you can still send it back."
        flushed = _flush_segment(text, first_segment=True)
        assert flushed == "The return window is thirty days, "

    def test_spanish_clause(self) -> None:
        text = "El plazo de devolución es de treinta días, así que puedes enviarlo hoy."
        flushed = _flush_segment(text, first_segment=True)
        assert flushed is not None
        assert flushed.rstrip().endswith("días,")
        assert "así" not in flushed

    def test_short_comma_waits_for_the_sentence(self) -> None:
        assert _flush_segment("Hello, how are you?", first_segment=True) == "Hello, how are you?"
        assert _flush_segment("Hola, ¿qué tal estás?", first_segment=True) == "Hola, ¿qué tal estás?"
        assert FIRST_SEGMENT_MIN_WORDS == 4

    def test_three_words_do_not_split_a_finished_sentence(self) -> None:
        text = "I can help, with the rest of this very long explanation today."
        assert _flush_segment(text, first_segment=True) == text
        assert _flush_segment("I can help, with the rest of this very long explanation", first_segment=True) is None

    def test_four_words_do_split(self) -> None:
        text = "I can help you, with the rest of this very long explanation today."
        flushed = _flush_segment(text, first_segment=True)
        assert flushed is not None
        assert flushed.rstrip().endswith("you,")

    def test_later_segments_ignore_the_comma(self) -> None:
        text = "The return window is thirty days, so you can still"
        assert _flush_segment(text, first_segment=False) is None

    def test_sentence_still_beats_a_later_comma(self) -> None:
        # No final period yet: this is the streaming path, where the first
        # sentence is already closed and the comma is still ahead of it.
        text = "Hello! The return window is thirty days, so you can still send it"
        flushed = _flush_segment(text, first_segment=True)
        assert flushed is not None
        assert flushed.rstrip().endswith("Hello!")
        assert "return" not in flushed

    def test_title_period_is_not_the_first_cut(self) -> None:
        assert _flush_segment("Ask Dr. Smith about the results today", first_segment=True) is None
        text = "La Sra. García llegará mañana por la tarde."
        assert _flush_segment(text, first_segment=True) == text
        streaming = "Ask Dr. Smith about the results today. Thanks for waiting"
        flushed = _flush_segment(streaming, first_segment=True)
        assert flushed is not None
        assert flushed.rstrip().endswith("today.")
        assert "Thanks" not in flushed
        # A finished reply with no long comma is still one segment.
        closed = "Ask Dr. Smith about the results today. Thanks for waiting."
        assert _flush_segment(closed, first_segment=True) == closed

    def test_thousands_separator_waits_for_lookahead(self) -> None:
        assert _flush_segment("The order number is 12,", first_segment=True) is None
        assert _flush_segment("The order number is 12,000 today and tomorrow", first_segment=True) is None
        text = "The order total is 12,000 euros, which covers shipping too."
        flushed = _flush_segment(text, first_segment=True)
        assert flushed is not None
        assert flushed.rstrip().endswith("euros,")
        assert "12,000" in flushed

    def test_audio_already_playing_does_not_rush_the_comma(self) -> None:
        text = "The return window is thirty days, so you can still send it back."
        assert _flush_segment(text, first_segment=True, audio_playing=True) is None

    def test_streaming_deltas_flush_once_the_next_word_arrives(self) -> None:
        deltas = ["The return window is thirty ", "days, so you can ", "still send it back."]
        buffer = ""
        scheduled = ""
        segments: list[str] = []
        for delta in deltas:
            buffer += delta
            flush = _flush_segment(buffer, first_segment=not scheduled)
            if flush is not None:
                segments.append(flush)
                scheduled += flush
                buffer = buffer[len(flush) :]
        if buffer:
            segments.append(buffer)
        assert segments == [
            "The return window is thirty days, ",
            "so you can still send it back.",
        ]
