"""Unit tests for VoiceSession STT duplicate/refinement handling."""

import unicodedata

from timbal.voice.session import (
    _flush_segment,
    _is_garbage_commit,
    _is_same_user_utterance_refinement,
    _pending_tts_after_scheduled,
    _reconcile_final_assistant_text,
)


def test_pending_tts_after_scheduled_first_sentence_only():
    first = "Hello! I'm doing great, thank you for asking."
    full = first + " How are you doing today? Is there anything I can help you with?"
    assert _pending_tts_after_scheduled(first, full) == " How are you doing today? Is there anything I can help you with?"


def test_pending_tts_after_scheduled_complete():
    full = "Hello! I'm doing great, thank you for asking. How are you doing today?"
    assert _pending_tts_after_scheduled(full, full) == ""


def test_pending_tts_mismatch_returns_empty():
    assert _pending_tts_after_scheduled("xyz", "abc") == ""


def test_pending_tts_nfc_stream_vs_message():
    """Streamed prefix and API ``Message`` can use different normalization for the same text."""
    prefix_nfc = "¡Hola! Muy bien, gracias por preguntar. ¿Y tú, cómo estás? ¿"
    full_nfc = prefix_nfc + "En qué te puedo ayudar hoy?"
    # Same graphemes as ``prefix_nfc`` but with a decomposed ó (o + combining acute).
    prefix_nfd = unicodedata.normalize("NFD", prefix_nfc)
    assert prefix_nfd != prefix_nfc
    assert _pending_tts_after_scheduled(prefix_nfd, full_nfc) == "En qué te puedo ayudar hoy?"


def test_reconcile_final_assistant_nfc_prefix():
    streamed = unicodedata.normalize("NFD", "Café ")
    final = "Café tail"
    merged, suffix = _reconcile_final_assistant_text(streamed, final)
    assert merged == final
    assert suffix == "tail"


def test_flush_segment_spanish_second_clause_ends_with_question():
    """Gemini often splits as a long ``Text`` + short ``text_delta``; tails can be < 40 chars."""
    tail = " tal estás? ¿En qué puedo ayudarte hoy?"
    assert _flush_segment(tail, first_segment=False) == tail


def test_flush_segment_very_short_second_clause_with_question_mark():
    """Short questions (< MIN_FLUSH_CHARS) that end with ? should flush immediately."""
    for tail in ("¿Sí?", " ¿Qué tal?", "¿Vale?"):
        assert len(tail) < 24  # MIN_FLUSH_CHARS
        assert _flush_segment(tail, first_segment=False) == tail


def test_flush_segment_splits_at_last_sentence_boundary():
    """Buffer with a sentence boundary in the middle should split there, not flush everything."""
    text = "¡Qué bueno escuchar eso! Me alegra mucho. ¿Hay algo en lo que te pueda"
    # first_segment=False → last boundary (greedy for prosody)
    result = _flush_segment(text, first_segment=False)
    assert result is not None
    assert result.rstrip().endswith("mucho.")
    assert "¿Hay algo" not in result


def test_flush_segment_first_segment_splits_at_earliest_qualifying_boundary():
    """first_segment=True splits at the earliest boundary that meets FIRST_SEGMENT_MIN_CHARS."""
    text = "¡Qué bueno escuchar eso! Me alegra mucho. ¿Hay algo en lo que te pueda"
    result = _flush_segment(text, first_segment=True)
    assert result is not None
    assert result.rstrip().endswith("eso!")
    assert "Me alegra" not in result


def test_flush_segment_complete_sentence_flushes_all():
    """Buffer ending on a sentence boundary flushes entirely."""
    text = "¡Qué bueno escuchar eso! Me alegra mucho."
    assert _flush_segment(text, first_segment=True) == text


def test_flush_segment_no_boundary_no_clause_end():
    """Buffer with no sentence boundary and no clause-ending char should not flush."""
    text = "Hay algo en lo que te pueda"
    assert _flush_segment(text, first_segment=False) is None


def test_flush_segment_audio_playing_buffers():
    """When audio is still playing, keep buffering instead of flushing eagerly."""
    text = "Hola, estoy muy bien. ¿En qué puedo ayudarte hoy?"
    # Without audio_playing → flushes at first boundary
    assert _flush_segment(text, first_segment=False) is not None
    # With audio_playing → keeps buffering (below MAX_TTS_BUFFER_CHARS)
    assert _flush_segment(text, first_segment=False, audio_playing=True) is None
    # But still flushes at MAX_TTS_BUFFER_CHARS even with audio_playing
    long_text = "A" * 201
    assert _flush_segment(long_text, first_segment=False, audio_playing=True) is not None


def test_flush_segment_first_segment_too_short():
    """First segment below FIRST_SEGMENT_MIN_CHARS should not flush."""
    assert _flush_segment("Hola", first_segment=True) is None
    assert _flush_segment("¡Hola!", first_segment=True) == "¡Hola!"


def test_flush_segment_inverted_question_not_clause_end():
    """¿ is an opening mark in Spanish — must NOT trigger clause-end flush.

    first_segment=True → earliest boundary → splits at "¡Hola! ".
    Subsequent split uses last boundary → ends at "estás? ", leaving only "¿".
    """
    text = "¡Hola! Muy bien, gracias por preguntar. ¿Y tú, cómo estás? ¿"
    result = _flush_segment(text, first_segment=True)
    assert result is not None
    assert result.rstrip() == "¡Hola!"

    remainder = text[len(result):]
    result2 = _flush_segment(remainder, first_segment=False)
    assert result2 is not None
    assert result2.rstrip().endswith("estás?")

    final_remainder = remainder[len(result2):]
    assert final_remainder == "¿"


def test_refinement_prefix_extension():
    a = "Hello, hello"
    b = "Hello, hello, how are you?"
    assert _is_same_user_utterance_refinement(a, b)


def test_refinement_duplicate():
    t = "Hello, hello, how are you?"
    assert _is_same_user_utterance_refinement(t, t)


def test_refinement_substring_when_long_enough():
    a = "Hello, hello, how are you"
    b = "Well hello, hello, how are you today?"
    assert len(a) >= 10
    assert _is_same_user_utterance_refinement(a, b)


def test_barge_in_shorter_not_refinement():
    active = "Hello, hello, how are you?"
    new = "stop"
    assert not _is_same_user_utterance_refinement(active, new)


def test_barge_in_unrelated_longer_not_refinement():
    active = "What is the weather"
    new = "Tell me a short story about space"
    assert not _is_same_user_utterance_refinement(active, new)


def test_garbage_lone_open_paren():
    assert _is_garbage_commit("(")
    assert _is_garbage_commit(" ( ")


def test_garbage_music_close_caption_hallucination():
    assert _is_garbage_commit("Music)")
    assert _is_garbage_commit("Applause)")


def test_garbage_incomplete_open_caption():
    assert _is_garbage_commit("(Music")
    assert _is_garbage_commit("(water splashing")


def test_garbage_not_real_utterances():
    assert not _is_garbage_commit("Probably, yeah, maybe you can tell me a story.")
    assert not _is_garbage_commit("no")
    assert not _is_garbage_commit("Hello")


def test_reconcile_final_assistant_empty_stream():
    merged, suffix = _reconcile_final_assistant_text("", "Hello world")
    assert merged == "Hello world"
    assert suffix == "Hello world"


def test_reconcile_final_assistant_prefix_stream():
    short = "I'm doing great!"
    full = "I'm doing great! How are you today?"
    merged, suffix = _reconcile_final_assistant_text(short, full)
    assert merged == full
    assert suffix == " How are you today?"


def test_reconcile_final_assistant_no_extension():
    t = "Same text"
    merged, suffix = _reconcile_final_assistant_text(t, t)
    assert merged == t
    assert suffix is None


def test_reconcile_missed_leading_text_block_tail_only():
    """Simulates only ``text_delta`` processed while final message has a prior clause."""
    tail = " Is there anything I can help you with?"
    full = (
        "I'm doing great, thank you for asking! How are you doing today?"
        + tail
    )
    merged, prefix = _reconcile_final_assistant_text(tail, full)
    assert merged == full
    assert prefix == "I'm doing great, thank you for asking! How are you doing today?"
