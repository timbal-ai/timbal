---
title: Memory
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Using Memory in Agents

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Learn how Agent handles memory and context through enhanced tracing and automatic memory resolution.
</h2>

---

## The Problem: Stateless AI Interactions

When building AI applications, you quickly encounter the challenge of maintaining context across interactions. Common problems include:

1. **Lost Context**: AI forgets previous conversations and user preferences
2. **Repetitive Information**: Users must repeat the same information in each interaction
3. **Inconsistent Behavior**: AI responses vary because they lack context
4. **Complex State Management**: Manual tracking of conversation history and user data
5. **Session Persistence**: Difficulty maintaining context across application restarts

Timbal's memory system solves these problems by providing automatic context management and persistent state.

## Memory in Agent

Agent handles memory through two primary mechanisms:

1. **Automatic Memory Resolution**: When agents are nested (used as tools by other agents), they automatically inherit conversation history
2. **Explicit State Management**: Memory is persistent across agent calls within the same execution session.

## Real-World Problem Solutions

### Problem 1: Customer Support Context Loss

**Challenge:** Customer support agents need to remember previous interactions, user preferences, and issue history.

**Solution:** Persistent conversation memory with user context

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent
from timbal.state import RunContext
from timbal.types.message import Message

# Create a customer support agent with memory
support_agent = Agent(
    name="customer_support",
    model="anthropic/claude-3-sonnet",
    system_prompt="""You are a customer support agent. Remember customer information and conversation history.
    Always refer to previous interactions when relevant."""
)

async def handle_customer_request(user_id: str, message: str):
    # Create or retrieve user context
    context = RunContext()
    context.data['user_id'] = user_id
    context.data['session_start'] = datetime.now().isoformat()
    
    # Load previous conversation history for this user
    previous_history = load_user_conversation_history(user_id)
    if previous_history:
        context.data['conversation_history'] = previous_history
    
    # Process the request with full context
    result = await support_agent(
        prompt=Message(role="user", content=message),
        context=context
    ).collect()
    
    # Save the updated conversation history
    save_user_conversation_history(user_id, result.messages)
    
    return result.output.content[0].text

# Example usage
response1 = await handle_customer_request("user123", "My name is Alice and I'm having trouble with my account")
response2 = await handle_customer_request("user123", "Can you help me reset my password?")
# The agent remembers Alice's name and account issues from the previous interaction
`}/>

### Problem 2: Multi-Step Workflow Context

**Challenge:** Complex workflows require maintaining state across multiple steps and agent interactions.

**Solution:** Workflow state management with automatic context passing

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent, Flow
from timbal.state import RunContext

# Define specialized agents for different workflow steps
data_analyst = Agent(
    name="data_analyst",
    model="anthropic/claude-3-sonnet",
    system_prompt="You are a data analyst. Use previous analysis results when available."
)

report_writer = Agent(
    name="report_writer", 
    model="openai/gpt-4",
    system_prompt="You are a report writer. Build on previous analysis and maintain consistency."
)

# Coordinator agent that manages the workflow
coordinator = Agent(
    name="workflow_coordinator",
    model="anthropic/claude-3-sonnet",
    tools=[data_analyst, report_writer],
    system_prompt="Coordinate analysis and reporting workflows. Maintain context across steps."
)

async def run_analysis_workflow(data_file: str, user_preferences: dict):
    # Create workflow context
    context = RunContext()
    context.data['workflow_type'] = 'data_analysis'
    context.data['data_file'] = data_file
    context.data['user_preferences'] = user_preferences
    context.data['workflow_start'] = datetime.now().isoformat()
    
    # Step 1: Data analysis
    analysis_result = await coordinator(
        prompt=f"Analyze the data in {data_file} according to user preferences: {user_preferences}",
        context=context
    ).collect()
    
    # Step 2: Report generation (context automatically passed)
    report_result = await coordinator(
        prompt="Generate a comprehensive report based on the analysis",
        context=context
    ).collect()
    
    return {
        'analysis': analysis_result.output.content[0].text,
        'report': report_result.output.content[0].text
    }
`}/>

### Problem 3: User Preference Management

**Challenge:** Different users have different preferences for AI behavior, language, and detail level.

**Solution:** Context-aware default parameters with user preferences

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent
from timbal.state import RunContext

async def user_preference_hook(input_dict):
    """Hook that sets parameters based on user preferences."""
    run_context = get_run_context()
    user_prefs = run_context.data.get('user_preferences', {})
    
    # Set response detail level
    if user_prefs.get('detailed_responses'):
        input_dict.setdefault('max_tokens', 2000)
        input_dict.setdefault('temperature', 0.1)
    else:
        input_dict.setdefault('max_tokens', 500)
        input_dict.setdefault('temperature', 0.7)
    
    # Set language preference
    language = user_prefs.get('language', 'english')
    if language == 'spanish':
        input_dict.setdefault('system_prompt', "Responde siempre en español.")
    elif language == 'french':
        input_dict.setdefault('system_prompt', "Répondez toujours en français.")
    else:
        input_dict.setdefault('system_prompt', "Always respond in English.")

# Create adaptive agent
adaptive_agent = Agent(
    name="adaptive_assistant",
    model="anthropic/claude-3-sonnet",
    pre_hook=user_preference_hook
)

async def handle_user_request(user_id: str, message: str):
    # Load user preferences
    user_prefs = load_user_preferences(user_id)
    
    # Create context with user preferences
    context = RunContext()
    context.data['user_preferences'] = user_prefs
    context.data['user_id'] = user_id
    
    # Agent automatically adapts to user preferences
    result = await adaptive_agent(
        prompt=message,
        context=context
    ).collect()
    
    return result.output.content[0].text
`}/>

## Automatic Memory Resolution

The most powerful feature of Agent is automatic memory resolution for nested agents:

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent
from timbal.types.message import Message

# Define a child agent that analyzes data
child_agent = Agent(
    name="data_analyst",
    model="anthropic/claude-3-sonnet",
    system_prompt="You are a data analyst. Analyze the provided data and give insights."
)

# Parent agent that coordinates tasks
parent_agent = Agent(
    name="coordinator", 
    model="openai/gpt-4",
    system_prompt="You coordinate different tasks and delegate to specialists.",
    tools=[child_agent]  # Child agent as a tool
)

# When the parent calls the child agent, memory is automatically shared
result = await parent_agent(
    prompt=Message(role="user", content="I have sales data for Q1. Please analyze it.")
).collect()

# The child agent automatically receives the full conversation context
# including the original user request and any previous exchanges
`}/>

### How Automatic Memory Works

When a child agent is called as a tool:

1. **Context Retrieval**: The child agent automatically retrieves the parent's conversation history
2. **Memory Reconstruction**: All previous messages (user inputs, assistant responses, tool calls) are included
3. **Seamless Continuation**: The child agent responds with full context awareness

<CodeBlock language="python" code ={`
# Behind the scenes, the child agent receives:
# [
#   Message(role="user", content="I have sales data for Q1. Please analyze it."),
#   Message(role="assistant", content="I'll analyze that data for you.", tool_calls=[...]),
#   Message(role="tool", content="Analysis results: ...")
# ]
`}/>

## Explicit State Management

For persistent memory across separate sessions, use RunContext:

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent
from timbal.state import RunContext
from timbal.types.message import Message

# Initialize agent
agent = Agent(
    name="memory_agent",
    model="openai/gpt-4"
)

# First interaction
result1 = await agent(
    prompt=Message(role="user", content="My name is Alice and I live in Paris")
).collect()

# Initialize a second agent with memory from the first
new_context = RunContext(parent_id=result1.run_id)
set_run_context(new_context)

# Second interaction with memory
result2 = await agent(
    prompt=Message(role="user", content="Where do I live?")
).collect()

print(result2.output)  # Should remember Alice lives in Paris
`}/>

## Advanced Memory Patterns

### Problem 4: Long Conversation Management

**Challenge:** Very long conversations exceed token limits and need intelligent summarization.

**Solution:** Conversation summarization and context compression

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent
from timbal.state import RunContext

async def summarize_conversation_hook(input_dict):
    """Hook that summarizes long conversations to stay within token limits."""
    run_context = get_run_context()
    conversation_history = run_context.data.get('conversation_history', [])
    
    # If conversation is too long, summarize it
    if len(conversation_history) > 20:  # More than 20 messages
        summary_agent = Agent(
            name="summarizer",
            model="gpt-4o-mini"
        )
        
        summary = await summary_agent(
            prompt=f"Summarize this conversation: {conversation_history[-10:]}"
        ).collect()
        
        # Replace long history with summary
        run_context.data['conversation_summary'] = summary.output.content[0].text
        run_context.data['conversation_history'] = conversation_history[-5:]  # Keep recent messages

# Agent with conversation management
long_conversation_agent = Agent(
    name="conversation_agent",
    model="anthropic/claude-3-sonnet",
    pre_hook=summarize_conversation_hook
)
`}/>

### Problem 5: Multi-User Session Management

**Challenge:** Managing multiple concurrent user sessions with different contexts.

**Solution:** Session-based context management

<CodeBlock language="python" code ={`
from timbal.core_v2 import Agent
from timbal.state import RunContext
import asyncio
from typing import Dict

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, RunContext] = {}
    
    def get_or_create_session(self, session_id: str) -> RunContext:
        if session_id not in self.sessions:
            self.sessions[session_id] = RunContext()
            self.sessions[session_id].data['session_id'] = session_id
            self.sessions[session_id].data['created_at'] = datetime.now().isoformat()
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, data: dict):
        if session_id in self.sessions:
            self.sessions[session_id].data.update(data)

# Global session manager
session_manager = SessionManager()

async def handle_multi_user_request(session_id: str, user_message: str):
    # Get or create session context
    context = session_manager.get_or_create_session(session_id)
    
    # Update session with user message
    session_manager.update_session(session_id, {
        'last_message': user_message,
        'last_activity': datetime.now().isoformat()
    })
    
    # Create agent with session context
    agent = Agent(
        name="multi_user_agent",
        model="anthropic/claude-3-sonnet"
    )
    
    # Process request with session context
    result = await agent(
        prompt=user_message,
        context=context
    ).collect()
    
    return result.output.content[0].text

# Handle multiple concurrent users
async def handle_concurrent_users():
    tasks = [
        handle_multi_user_request("user1", "Hello, I'm Alice"),
        handle_multi_user_request("user2", "Hi, I'm Bob"),
        handle_multi_user_request("user1", "What's my name?")  # Should remember Alice
    ]
    
    results = await asyncio.gather(*tasks)
    return results
`}/>

## Memory with System Prompts

Agent's system prompt works seamlessly with memory:

<CodeBlock language="python" code ={`
agent = Agent(
    name="context_aware_agent",
    model="anthropic/claude-3-sonnet",
    system_prompt="You are a helpful assistant with access to conversation history."
)

# First interaction
result1 = await agent(
    prompt=Message(role="user", content="I'm working on a Python project")
).collect()

# Second interaction with memory
result2 = await agent(
    prompt=Message(role="user", content="Can you help me debug this code?")
).collect()

# The agent remembers the Python project context from the system prompt
`}/>

## Best Practices

1. **Use Descriptive Context Keys**: Use clear, descriptive keys for context data
2. **Implement Context Cleanup**: Regularly clean up old context data to prevent memory bloat
3. **Validate Context Data**: Always validate context data before using it
4. **Handle Context Errors**: Implement fallback behavior when context is unavailable
5. **Monitor Context Size**: Track context size to avoid token limit issues
6. **Test Memory Behavior**: Thoroughly test memory behavior across different scenarios

## Summary

Timbal's memory management provides:

- **Automatic Memory Resolution**: Seamless context sharing for nested agents
- **Enhanced State Management**: Improved RunContext and state saver integration  
- **Flexible Memory Patterns**: Custom memory resolution and filtering
- **Performance Optimization**: Token-aware memory compression
- **Rich Debugging**: Comprehensive tracing and logging

The architecture makes memory handling more robust and automatic while providing advanced customization options for complex use cases.