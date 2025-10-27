import os

from dotenv import load_dotenv
from slack_sdk import WebClient
from timbal import Agent
from timbal.errors import bail
from timbal.state import get_run_context

# Load environment variables from .env
load_dotenv()

SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

async def pre_hook():
    """Process incoming Slack webhook events before agent execution."""
    span = get_run_context().current_span()

    # Only process Slack webhook events
    # Note: "_webhook" must match the parameter name used when calling agent(_webhook=body)
    if "_webhook" not in span.input:
        return

    slack_event = span.input["_webhook"]["event"]

    # Prevent infinite loops by ignoring bot's own messages
    if slack_event.get("user") == SLACK_BOT_USER_ID:
        raise bail()

    # Extract message text
    text = slack_event.get("text", "")
    if not text:
        raise bail()

    # Store context for response
    span.slack_channel = slack_event["channel"]
    span.slack_thread_ts = slack_event.get("thread_ts")
    span.input["prompt"] = text

def post_hook():
    """Send agent response back to Slack."""
    span = get_run_context().current_span()
    
    if "_webhook" not in span.input:
        return

    # Get response and send to Slack
    reply = span.output.content[0].text
    slack_channel = span.slack_channel
    slack_thread_ts = span.slack_thread_ts
    
    if reply.strip():
        client.chat_postMessage(
            channel=slack_channel,
            text=reply,
            thread_ts=slack_thread_ts,
        )

# Create the Slack agent
agent = Agent(
    name="SlackAgent",
    system_prompt="You are a tech support AI assistant. Help users with their technical questions clearly and concisely.",
    model="openai/gpt-4.1-mini",
    pre_hook=pre_hook,
    post_hook=post_hook,
)