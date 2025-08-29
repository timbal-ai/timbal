---
title: Advanced
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Advanced Agent Concepts

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master advanced patterns for Agent including memory management, nested execution, custom schemas, and performance optimization.
</h2>

---

### Execution Hooks

You can use it either in `Agent` or `Tool`.

Use hooks to clean input data and handle outputs.

We can see one example by a Slack configuration that receives messages from slack and has to send via it the result.

<CodeBlock language="python" code ={`def process_message(message: str) -> str:
    """Process user message."""
    return f"Processed: {message}"

async def clean_slack_input(input_data: dict) -> None:
    """Clean Slack formatting before passing to LLM."""
    message = input_data["message"]
    # Remove Slack mentions and formatting
    clean_message = message.replace("<@U123>", "").replace("*", "").strip()
    input_data["message"] = clean_message

async def send_to_slack(output: str) -> None:
    """Send result back to Slack channel."""
    # Format and send response
    slack_client.send_message(channel="#general", text=output)
    print(f"Sent to Slack: {output}")

agent = Agent(
    handler=process_message,
    pre_hook=clean_slack_input,   # Clean input from Slack
    post_hook=send_to_slack       # Send result to Slack
)`}/>


### Timbal Platform Integration

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

