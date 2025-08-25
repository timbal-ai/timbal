---
title: Using Memory
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Using Memory in Workflows

:::warning[Work in Progress]
The Workflow class is currently under development and not fully implemented. The examples in this documentation may not work as expected.
:::

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Understand workflow state persistence and context management for building stateful and conversational workflows.
</h2>

---

When we talk about "memory" in Timbal workflows, we refer to the ability to maintain state and context across workflow executions.

## Workflow State Management

Workflows can maintain state across executions using the state management system. This allows you to:

- **Persist data**: Save information between workflow runs
- **Track context**: Maintain conversation history and user sessions
- **Debug workflows**: Analyze execution history and state changes
- **Resume workflows**: Continue from where they left off

## Using Run Context

The primary way to manage state in workflows is through the run context, which is automatically available to all steps.

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.state import get_run_context

def counter_step() -> int:
    context = get_run_context()
    current_count = context.get("counter", 0)
    new_count = current_count + 1
    context["counter"] = new_count
    return new_count

def get_total_count() -> int:
    context = get_run_context()
    return context.get("counter", 0)

workflow = (
    Workflow(name="stateful_counter")
    .add_step(counter_step)
    .add_step(counter_step)
    .add_step(get_total_count)
)

async def main():
    result1 = await workflow.complete()  # Counter: 1, 2, 3
    result2 = await workflow.complete()  # Counter: 4, 5, 6
    print(f"First run total: {result1.output}")
    print(f"Second run total: {result2.output}")
`}/>

## Persistent State with State Savers

For more advanced state management, you can use state savers to persist workflow state to external storage.

### In-Memory State Saver

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.state.savers import InMemorySaver

def user_preference_step() -> str:
    context = get_run_context()
    preferences = context.get("user_preferences", [])
    preferences.append("dark_mode")
    context["user_preferences"] = preferences
    return f"Preferences: {preferences}"

workflow = (
    Workflow(name="user_preferences")
    .add_step(user_preference_step)
)

# Use in-memory state saver for testing
workflow_with_state = workflow.compile(state_saver=InMemorySaver())

async def main():
    result1 = await workflow_with_state.complete()
    result2 = await workflow_with_state.complete()
    # State persists between runs
`}/>

### File-Based State Saver

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.state.savers import JSONLSaver

def session_tracker() -> dict:
    context = get_run_context()
    session_count = context.get("session_count", 0) + 1
    context["session_count"] = session_count
    return {"session": session_count, "timestamp": "2024-01-01"}

workflow = (
    Workflow(name="session_tracker")
    .add_step(session_tracker)
)

# Save state to file
workflow_with_persistence = workflow.compile(state_saver=JSONLSaver("sessions.jsonl"))

async def main():
    # State will be saved to sessions.jsonl
    result = await workflow_with_persistence.complete()
`}/>

## Conversation Memory

For conversational workflows, you can maintain conversation history and context:

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.state import get_run_context

def chat_step(user_message: str) -> str:
    context = get_run_context()
    
    # Get conversation history
    history = context.get("conversation_history", [])
    
    # Add user message to history
    history.append({"role": "user", "content": user_message})
    
    # Simulate AI response
    ai_response = f"AI: I understand you said '{user_message}'. This is response #{len(history)}."
    history.append({"role": "assistant", "content": ai_response})
    
    # Save updated history
    context["conversation_history"] = history
    
    return ai_response

def get_conversation_summary() -> str:
    context = get_run_context()
    history = context.get("conversation_history", [])
    return f"Conversation has {len(history)} messages"

workflow = (
    Workflow(name="conversational_ai")
    .add_step(chat_step, user_message="Hello")
    .add_step(get_conversation_summary)
)

async def main():
    result1 = await workflow.complete()  # First conversation
    result2 = await workflow.complete()  # Second conversation (continues from first)
`}/>

## State Isolation

You can create isolated state contexts for different workflow instances:

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.state import get_run_context

def user_session_step(user_id: str) -> dict:
    context = get_run_context()
    
    # Create user-specific state
    user_state = context.get("users", {})
    if user_id not in user_state:
        user_state[user_id] = {"visits": 0, "preferences": []}
    
    user_state[user_id]["visits"] += 1
    context["users"] = user_state
    
    return user_state[user_id]

workflow = (
    Workflow(name="multi_user_tracker")
    .add_step(user_session_step, user_id="user1")
    .add_step(user_session_step, user_id="user2")
)

async def main():
    result = await workflow.complete()
    # Each user has their own isolated state
`}/>

## Debugging with State

State management also helps with debugging workflows:

<CodeBlock language="python" code={`from timbal.core_v2 import Workflow
from timbal.state import get_run_context

def debug_step(data: dict) -> dict:
    context = get_run_context()
    
    # Log state changes
    debug_info = context.get("debug_log", [])
    debug_info.append({
        "step": "debug_step",
        "input": data,
        "timestamp": "2024-01-01T12:00:00Z"
    })
    context["debug_log"] = debug_info
    
    return {"processed": data, "debug_count": len(debug_info)}

workflow = (
    Workflow(name="debuggable_workflow")
    .add_step(lambda: {"test": "data"})
    .add_step(debug_step)
    .add_step(debug_step)
)

async def main():
    result = await workflow.complete()
    # Debug log contains execution history
`}/>

## Best Practices

### State Key Naming

Use descriptive, namespaced keys to avoid conflicts:

<CodeBlock language="python" code={`def good_state_management():
    context = get_run_context()
    
    # Good: Namespaced keys
    context["user_preferences.theme"] = "dark"
    context["user_preferences.language"] = "en"
    
    # Good: Structured data
    context["session_data"] = {
        "user_id": "123",
        "start_time": "2024-01-01T12:00:00Z",
        "actions": []
    }
`}/>

### State Cleanup

Consider cleaning up old state to prevent memory bloat:

<CodeBlock language="python" code={`def cleanup_old_state():
    context = get_run_context()
    
    # Keep only recent history
    history = context.get("conversation_history", [])
    if len(history) > 100:
        context["conversation_history"] = history[-50:]  # Keep last 50 messages
`}/>

---

For more, see the [Workflows Overview](/workflows) and [Advanced Workflow Concepts](/workflows/advanced).
