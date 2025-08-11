---
title: Voice
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Voice: Speech Capabilities for Agents

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Add speech-to-text and text-to-speech capabilities to your agents with multiple voice providers.
</h2>

---

Timbal agents can understand and generate speech using a variety of voice providers. This enables use cases like voice assistants, audio chatbots, and speech-to-speech interactions.

## Using a Single Provider

You can add voice capabilities to your agent using a single provider, such as OpenAI or ElevenLabs.

<span style={{color: 'var(--timbal-purple)'}}><strong>Speech-to-Text (STT):</strong></span> Convert audio to text.

<CodeBlock language="python" code={`from timbal.steps.elevenlabs import stt
from timbal.types import File

audio_file = File.validate("path/to/audio.wav")
transcription = await stt(audio_file=audio_file)
print(transcription)`}/>

<span style={{color: 'var(--timbal-purple)'}}><strong>Text-to-Speech (TTS):</strong></span> Convert text to audio.

<CodeBlock language="python" code={`from timbal.steps.elevenlabs import tts

audio_file = await tts(
    text="Hello, how are you?",
    voice_id="your-voice-id"
)
# audio_file is a File object containing the generated audio`}/>

---

## Using Multiple Providers

You can mix and match providers for STT and TTS. For example, use OpenAI for transcription and ElevenLabs for speech generation.

<CodeBlock language="python" code={`from timbal.steps.openai import stt as openai_stt
from timbal.steps.elevenlabs import tts as elevenlabs_tts
from timbal.types import File

# Transcribe with OpenAI
audio_file = File.validate("https://cdn.openai.com/API/docs/audio/alloy.wav")
audio_text = await openai_stt(audio_file=audio_file)

# Synthesize with ElevenLabs
audio_response = await elevenlabs_tts(
        text=audio_text, 
        voice_id="21m00Tcm4TlvDq8ikWAM"
)`}/>

---

## Working with Audio Streams

You can pass audio files directly as prompts to agents. The agent will automatically use the appropriate STT tool if available.

<CodeBlock language="python" code={`from timbal import Agent
from timbal.steps.elevenlabs import stt
from timbal.types import File

agent = Agent(
    tools=[stt]
)

audio_file = File.validate("https://cdn.openai.com/API/docs/audio/alloy.wav")
response = await agent.complete(prompt=audio_file)
print(response.output.content[0].text)`}/>

---

## Speech-to-Speech Voice Interactions

You can build agents that both understand and respond in audio. For example, an agent that receives audio, transcribes it, generates a response, and then synthesizes speech:

<CodeBlock language="python" code={`from timbal import Agent
from timbal.steps.elevenlabs import stt, tts

agent = Agent(
    tools=[{
        "runnable": tts,
        "force_exit": True
    }],
    system_prompt=(
        "You are a helpful assistant and you must always respond in audio format."
        "Always use '56AoDkrOh6qfVPDXZ7Pt' as the voice_id for the TTS model."
    )
)

response = await agent.complete(prompt="How are you?")
# response.output will be a File (audio) if TTS is used`}/>


<div className="log-step-static">
StartEvent(..., path='agent', ...)
</div>
<div className="log-step-static">
StartEvent(..., path='agent.llm-0', ...)
</div>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='agent.llm-0', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(..., 
    path='agent.llm-0', 
    input={
    'messages': [
        Message(
            role=user, 
            content=[TextContent(type='text', text='How are you?')]
        )
    ], 
    'tools': [{
        'type': 'function', 
        'function': {
            'name': 'tts', 
            'description': '', 
            'parameters': {
                'properties': {
                    'text': {
                        'description': 'The text to convert to speech.', 
                        'title': 'Text', 'type': 'string'}, 
                        'voice_id': {'description': 'The voice ID to use for text-to-speech.', 
                        'title': 'Voice Id', 'type': 'string'}, 
                        'model_id': {'default': 'eleven_flash_v2_5', 'description': 'The model to use for text-to-speech.', 'enum': ['eleven_flash_v2_5', 'eleven_multilingual_v2'], 'title': 'Model Id', 'type': 'string'}
                    }, 
                    'required': ['text', 'voice_id'], 
                    ...
                }
            }
        }], 
        'system_prompt': "You are a helpful assistant and you must always respond in audio format. Always use '56AoDkrOh6qfVPDXZ7Pt' as the voice_id for the TTS model.", 
        'model': 'gpt-4o-mini', ...
    }, 
    output=Message(
        role=assistant,
        content=[ToolUseContent(type='tool_use', id='call_...', name='tts', input={'text': "I'm just a program, so I don't have feelings, but I'm here ready to assist you! How can I help you today?", 'voice_id': '56AoDkrOh6qfVPDXZ7Pt'})]
    ), ...
)`}/>
</details>
<div className="log-step-static">
StartEvent(..., path='agent.tts-call_...', ...)
</div>
<details className="log-step-collapsible">
<summary>
    OutputEvent(..., path='agent.tts-call_...', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(..., 
    path='agent.tts-call_...',
    input={
        'text': "I'm just a program, so I don't have feelings, but I'm here ready to assist you! How can I help you today?",
        'voice_id': '56AoDkrOh6qfVPDXZ7Pt',
        'model_id': 'eleven_flash_v2_5'
    },
    output=Message(
        role=user,
        content=[FileContent(type='file', file=File(source=io.IOBase(.mp3)))]
    ), ...
)`}/>
</details>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='agent', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(..., 
    path='agent', 
    input={
        'prompt': {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'How are you?'}]
        }
    },
    output=Message(
        role=user,
        content=[FileContent(type='file', file=File(source=io.IOBase(.mp3)))]
    ), ...
)`}/>
</details>

---

## Supported Voice Providers

Timbal supports multiple providers for both STT and TTS:

- **OpenAI**: High-quality transcription and speech synthesis.
- **ElevenLabs**: Advanced, natural-sounding voices and robust transcription.
- (More providers coming soon!)

---

For more details, see the [ElevenLabs Integration](/integrations/elevenlabs) and [OpenAI Integration] pages.
