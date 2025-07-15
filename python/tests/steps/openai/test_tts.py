import os
import pytest
from pathlib import Path

from timbal.steps.openai.tts import tts
from timbal.types import File
from timbal.errors import APIKeyNotFoundError


# Integration tests that require real API keys
# These tests are marked separately so they can be skipped if no API key is available

@pytest.mark.asyncio
async def test_real_tts_basic():
    """Integration test for basic TTS functionality with real API."""
    
    # Skip if no API key is available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")
    
    result = await tts(text="Hello, this is a test of OpenAI text-to-speech.")
    
    # Verify the result
    assert isinstance(result, File)
    assert result.__content_type__.startswith("audio/")
    assert result.__source_extension__ == ".mp3"
    
    # Read content only once and check
    content = result.read()
    assert len(content) > 0  # Should have audio content
    assert len(content) > 100  # Audio files should be reasonably sized


@pytest.mark.asyncio
async def test_real_tts_different_voice():
    """Integration test with different voice."""
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")
    
    result = await tts(
        text="This is a test with the Nova voice.",
        voice="nova"
    )
    
    assert isinstance(result, File)
    assert result.__content_type__.startswith("audio/")
    content = result.read()
    assert len(content) > 100


@pytest.mark.asyncio
async def test_real_tts_wav_format():
    """Integration test with WAV format."""
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")
    
    result = await tts(
        text="Testing WAV format output.",
        response_format="wav"
    )
    
    assert isinstance(result, File)
    assert result.__source_extension__ == ".wav"
    content = result.read()
    assert len(content) > 100
    # WAV files start with "RIFF" header
    assert content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_real_tts_gpt4o_mini_with_instructions():
    """Integration test with gpt-4o-mini-tts model and instructions."""
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")
    
    result = await tts(
        text="This message should sound very cheerful and enthusiastic!",
        model_id="gpt-4o-mini-tts",
        instructions="Speak in a very cheerful, enthusiastic, and upbeat tone."
    )
    
    assert isinstance(result, File)
    assert result.__content_type__.startswith("audio/")
    content = result.read()
    assert len(content) > 100


@pytest.mark.asyncio
async def test_real_tts_long_text():
    """Integration test with longer text content."""
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")
    
    long_text = """
    This is a longer text to test the text-to-speech functionality with more content.
    The OpenAI TTS API should be able to handle longer passages of text without issues.
    This test helps ensure that the implementation works correctly with realistic use cases
    where users might want to convert entire paragraphs or articles to speech.
    """
    
    result = await tts(text=long_text.strip())
    
    assert isinstance(result, File)
    assert result.__content_type__.startswith("audio/")
    content = result.read()
    # Longer text should produce longer audio
    assert len(content) > 1000


@pytest.mark.asyncio 
async def test_real_tts_multiple_formats():
    """Integration test to verify different output formats work."""
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")
    
    text = "Testing different audio formats."
    formats_to_test = ["mp3", "wav", "opus"]
    
    results = {}
    
    for format_name in formats_to_test:
        result = await tts(
            text=text,
            response_format=format_name
        )
        
        assert isinstance(result, File)
        results[format_name] = result
        
        # Verify correct extension
        expected_extension = f".{format_name}"
        assert result.__source_extension__ == expected_extension
        
        # Verify has content
        content = result.read()
        assert len(content) > 100
    
    # Different formats should produce different file sizes/content
    # (though this isn't guaranteed, it's likely)
    mp3_content = results["mp3"].read()
    wav_content = results["wav"].read()
    
    # WAV is typically uncompressed so should be larger
    assert len(wav_content) >= len(mp3_content)
