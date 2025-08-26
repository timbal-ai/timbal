---
title: ElevenLabs
sidebar: 'docsSidebar_v2'
description: 'Integrate ElevenLabs speech-to-text and text-to-speech capabilities into your Timbal workflows'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# ElevenLabs Integration

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Convert speech to text and generate natural-sounding speech with ElevenLabs' advanced AI voice models.
</h2>

---

Timbal provides seamless integration with ElevenLabs' powerful AI voice and speech services. 

This integration allows you to:
- Convert speech to text using ElevenLabs' advanced transcription models
- Generate natural-sounding speech from text using various voices and models

## Prerequisites

Before using the ElevenLabs integration, you'll need:

1. An ElevenLabs account - [Sign up here](https://elevenlabs.io/app/sign-in)
2. An API key - [Get your API key here](https://elevenlabs.io/app/settings/api-keys)
3. Set up your environment variable:
   <CodeBlock language="bash" code ={`export ELEVENLABS_API_KEY='your-api-key-here`}/>

## Installation

No additional installation is required.

Import the specific functions you need:

<CodeBlock language="python" code ={`from timbal.steps.elevenlabs import stt, tts`}/>

1. Sign up for a ElevenLabs account [here](https://elevenlabs.io/app/sign-in), and obtain an [API key](https://elevenlabs.io/app/settings/api-keys).
2. Store your obtained API key in an environment variable named `ELEVENLABS_API_KEY` to facilitate its use by the tools.


## <span style={{color: 'var(--timbal-purple)'}}><strong>Speech to Text (STT)</strong></span>

### Description
The Speech to Text (STT) service converts audio files into text using ElevenLabs' advanced transcription models.

### Example
<CodeBlock language="python" code ={`from timbal.steps.elevenlabs import stt
from timbal.types import File

# Validate and process an audio file
audio_file = File.validate("path/to/audio.wav")
transcription = await stt(audio_file=audio_file, model_id="scribe_v1")`}/>


### Parameters

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>audio_file</code></td>
      <td><code>File</code></td>
      <td>Audio file to transcribe. Must be a valid audio file (content type starting with "audio/")</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>model_id</code></td>
      <td><code>str</code></td>
      <td>Transcription model to use. Available models:<br/>- `scribe_v1`: Standard transcription model<br/>- `scribe_v1_experimental`: Experimental model with enhanced features</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>



<CodeBlock language="python" code ={`from timbal.steps.elevenlabs import stt
from timbal.types import File
from timbal import Agent

# Create an agent with STT capability
agent = Agent(
    tools=[stt]
)

# Process an audio file
audio_file = File.validate("path/to/audio.wav")
response = await agent.complete(prompt=audio_file, "What does it say?")`}/>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Text to Speech (TTS)</strong></span>

The Text to Speech (TTS) service converts text into natural-sounding speech using ElevenLabs' voice models.

### Example
<CodeBlock language="python" code ={`from timbal.steps.elevenlabs import tts

# Generate speech from text
audio_file = await tts(
    text="Hello, how are you?",
    voice_id="your-voice-id",
    model_id="eleven_flash_v2_5"
)`}/>

### Parameters

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>text</code></td>
      <td><code>str</code></td>
      <td>Text to convert to speech</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>voice_id</code></td>
      <td><code>str</code></td>
      <td>ID of the voice to use</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>model_id</code></td>
      <td><code>str</code></td>
      <td>TTS model to use. Available models:<br/>- `eleven_flash_v2_5`: Fast and efficient model<br/>- `eleven_multilingual_v2`: Model with multilingual support</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>


### Integration with Agent

<CodeBlock language="python" code ={`from timbal.steps.elevenlabs import tts
from timbal import Agent
from timbal.types.message import Message

# Create an agent that responds with audio
agent = Agent(
      name="audio_agent",
      model="openai/gpt-4o",
      system_prompt="Answer always with an audio."
      tools=[tts]
)

response = await agent(prompt="What is 2+2?").collect()`}/>