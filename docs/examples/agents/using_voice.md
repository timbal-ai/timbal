---
title: Using Voice with Agents
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Giving your Agent a Voice

Timbal agents can be enhanced with voice capabilities, enabling them to speak and listen. This example demonstrates how to configure voice functionality for your agents.

## Prerequisites

This example uses the `openai` model. Make sure to add `OPENAI_API_KEY` to your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

## Voice Tools

Create TTS and STT tools that the agent can use to process audio input and generate audio responses:

### Speech-to-Text Tool

<CodeBlock language="python" code={`from timbal.core import Tool
from timbal.handlers.openai.stt import stt

# Create the STT tool
stt_tool_instance = Tool(
    name="speech_to_text",
    description="Convert audio input to text.",
    handler=stt
)`}/>

### Text-to-Speech Tool

<CodeBlock language="python" code={`from timbal.core import Tool
from timbal.handlers.openai.tts import tts

# Create the TTS tool
tts_tool_instance = Tool(
    name="text_to_speech",
    description="Convert text to speech.",
    handler=tts
)`}/>

## Voice-Enabled Agent

Create an agent with voice tools that can read audio files and respond with audio:

<CodeBlock language="python" code={`from timbal.core import Agent

voice_agent = Agent(
    name="voice-agent",
    description="An agent with voice capabilities for speaking and listening",
    system_prompt="""You are a voice-enabled agent that MUST follow:
1. Convert the audio input to text using stt_tool_instance
2. Process the text and generate a response
3. Convert your response to speech using the tts_tool_instance

Always use the speech_to_text tool when you receive audio input, and use the tts_tool_instance tool to respond with audio.""",
    model="openai/gpt-4.1",
    tools=[stt_tool_instance, tts_tool_instance]
)`}/>

## Example usage

This example shows how the agent can process audio input and respond with audio:

<CodeBlock language="python" code={`import asyncio
from timbal.types.file import File

async def main():
    # Example: Agent receives audio input and responds with audio
    
    # 1. Create a message with audio input
    # Use a reliable sample audio file for testing
    audio_file = File.validate("https://cdn.openai.com/API/docs/audio/alloy.wav")
    
    prompt = [audio_file, "Please listen to this audio and respond with speech."]

    # 2. Agent processes audio and responds with audio
    response = await voice_agent(prompt=prompt).collect()
    
    # 3. The response will contain audio content
    print("Agent processed audio input and generated audio response!")
    
    # 4. Save the audio response
    if response.output.content:
        for content in response.output.content:
            if hasattr(content, 'file') and content.file:
                output_path = "agent_response.mp3"
                content.file.to_disk(output_path)
                print(f"Audio response saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())`}/>


## Available voice handlers

Timbal provides several built-in voice handlers:

### OpenAI Handlers
- **STT**: `timbal.handlers.openai.stt.stt` - [OpenAI Integration](/)
- **TTS**: `timbal.handlers.openai.tts.tts` - [OpenAI Integration](/)

### ElevenLabs Handler
- **STT**: `timbal.handlers.elevenlabs.stt.stt` - [ElevenLabs Integration](/)
- **TTS**: `timbal.handlers.elevenlabs.tts.tts` - [ElevenLabs Integration](/)

### Configuration
Make sure you have the required API keys set in your environment:
- `OPENAI_API_KEY` for OpenAI voice services
- `ELEVENLABS_API_KEY` for ElevenLabs services

The agent now has voice tools as part of its capabilities, making it truly voice-enabled!

<div>
  <Link className={styles.card} href="https://github.com/your-repo/design-tools" target="_blank" style={{display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '1.2rem', flexWrap: 'nowrap'}}>
    <span className={styles.icon}><svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg></span>
    <span style={{flexShrink: 0}}>Using Voice</span>
    <span style={{flexShrink: 0, marginLeft: 'auto', fontSize: '1.5rem'}}>↗</span>
  </Link>
</div> 