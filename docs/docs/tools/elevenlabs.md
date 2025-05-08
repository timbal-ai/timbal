---
title: ElevenLabs
sidebar: 'docsSidebar'
---

# ElevenLabs Tool

## <span style={{color: 'var(--timbal-purple)'}}><strong>Speech to Text (STT)</strong></span>

```python
async def stt(
    audio_file: File = Field(
        description=(
            "The audio file to transcribe. "
            "All major audio and video formats are supported. "
            "The file size must be less than 1GB."
        ),
    ),
    model_id: str = Field(
        default="scribe_v1", 
        description="The elevenlabs model to use for the STT.",
        choices=["scribe_v1", "scribe_v1_experimental"],
    ),
) -> str
```

Converts an audio file to text using ElevenLabs' speech-to-text service.

### Parameters
- `audio_file`: The audio file to transcribe. Must be an audio file with content type starting with "audio/".
- `model_id`: The model to use for transcription. Options:
  - `"scribe_v1"`: Standard model
  - `"scribe_v1_experimental"`: Experimental model

### Returns
- `str`: The transcribed text from the audio file.

### Requirements
- `ELEVENLABS_API_KEY` environment variable must be set with a valid ElevenLabs API key.

### Example
```python
from timbal.steps.elevenlabs import stt
from timbal.types import File

audio_file = File.validate("path/to/audio.wav")
transcription = await stt(audio_file=audio_file, model_id="scribe_v1")
```

## <span style={{color: 'var(--timbal-purple)'}}><strong>Text to Speech (TTS)</strong></span>

```python
async def tts(
    text: str = Field(
        description="The text to convert to speech.",
    ),
    voice_id: str = Field(
        description="The voice ID to use for text-to-speech.",
    ),
    model_id: str = Field(
        default="eleven_flash_v2_5",
        description="The model to use for text-to-speech.",
        choices=["eleven_flash_v2_5", "eleven_multilingual_v2"],
    ),
) -> File
```

Converts text to speech using ElevenLabs' text-to-speech service.

### Parameters
- `text`: The text to convert to speech.
- `voice_id`: The ID of the voice to use for speech generation.
- `model_id`: The model to use for speech generation. Options:
  - `"eleven_flash_v2_5"`: Fast and efficient model
  - `"eleven_multilingual_v2"`: Multilingual support model

### Returns
- `File`: An MP3 audio file containing the generated speech.

### Requirements
- `ELEVENLABS_API_KEY` environment variable must be set with a valid ElevenLabs API key.

### Example
```python
from timbal.steps.elevenlabs import tts

audio_file = await tts(
    text="Hello, how are you?",
    voice_id="your-voice-id",
    model_id="eleven_flash_v2_5"
)
```