---
title: Advanced
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Advanced Agent Concepts

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master advanced patterns for Agent including memory management, nested execution, custom schemas, and performance optimization.
</h2>

---

## Execution Hooks

Hooks process data before (`pre_hook`) and after (`post_hook`) agent execution. Available for both `Agent` and `Tool`.

### Example: Slack Integration

Clean incoming Slack messages and auto-send responses.

:::info
Import `get_run_context` from Timbal. See Slack Documentation (integrations_v2/slack) for `send_message` and other functions.
:::

**Pre-hook:** Clean Slack message formatting:
<CodeBlock language="python" code ={`async def pre_hook():
    slack_messages = get_run_context().get_data(".input.prompt")
    text = slack_messages[0].get("text", "")
    clean_text = re.sub(r'@[A-Z0-9]+', '', text)
    clean_text = clean_text.replace("*", "").replace("_", "").strip()
    get_run_context().set_data(".input.prompt", clean_text)`}/>

**Post-hook:** Auto-send agent response to Slack:

<CodeBlock language="python" code ={`async def post_hook():
    output = get_run_context().get_data(".output")["content"][0]["text"]
    send_message(channel="channel-id", text=output)`}/>

**Complete Agent setup:**

<CodeBlock language="python" code ={`Agent(
    name="slack_agent",
    model="openai/gpt-4.1-mini",
    pre_hook=pre_hook,   # Process Slack message before agent
    post_hook=post_hook, # Handle response after agent
    system_prompt="You are a helpful assistant. Respond concisely and friendly to Slack messages."
)`}/>

Hooks provide a powerful way to add middleware functionality to your Timbal components, enabling input/output transformation, validation, monitoring, and context-aware behavior.


### Example: Voice Integration

You can create voice agents that receive audio input and return audio output using pre and post hooks. This enables speech-to-speech interactions where users speak to the agent and receive spoken responses.


<CodeBlock language="python" code={`from timbal import Agent
from timbal.handlers.openai.stt import stt
from timbal.handlers.openai.tts import tts
from timbal.types import File
from timbal.state import get_run_context

async def audio_pre_hook():
  audio_input = get_run_context().get_data(".input")
  # You need to have a reference to the object, to then modify the key directly:
  audio_input["prompt"] = await stt(audio_file=audio_input["prompt"])
  get_run_context().set_data(".input", audio_input)

async def audio_post_hook():
  audio_output = get_run_context().get_data(".output")
  
  output_with_audio = {  # Use a different variable name
    "text": audio_output.content[0].text,
    "audio": await tts(text=audio_output.content[0].text)
  }
  get_run_context().set_data(".output", output_with_audio)

# Create agent with voice tools
voice_agent = Agent(
  name="voice_assistant",
  model="openai/gpt-4o-mini",
  pre_hook=audio_pre_hook,
  post_hook=audio_post_hook,
)

# Send audio input
audio_input = File.validate("path/to/user_audio.wav")
response = await voice_agent(prompt=audio_input).collect()

# The agent will automatically use STT to transcribe input and TTS to generate audio output`}/>

This approach gives the agent more control over when and how to use voice processing, while the pre/post-hook approach provides automatic voice conversion for all interactions.


## Timbal Platform Integration

:::warning
This should be in Platform Section
:::

By default, agents save conversation memory locally. The Timbal Platform provides centralized memory storage and comprehensive tracing capabilities.

**Platform Benefits:**
- Persistent memory across deployments
- Complete conversation tracing and analytics
- Real-time monitoring without rebuilding

**Environment Variables:**

Set these variables to enable platform integration without deploying a new version:

<CodeBlock language="bash" code ={`TIMBAL_API_HOST=https://api.timbal.ai
TIMBAL_API_KEY=your_api_key
TIMBAL_APP_ID=your_app_id
TIMBAL_ORG_ID=your_org_id`}/>

