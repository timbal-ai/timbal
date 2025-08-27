---
title: Memory
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Using Memory in Agents

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Learn how Agent handles memory and context through enhanced tracing and automatic memory resolution.
</h2>

---

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

### Example 1: Simple conversation memory

<CodeBlock language="python" code ={`from timbal.core import Agent

# Agents automatically maintain conversation memory
agent = Agent(
    name="chatbot",
    model="anthropic/claude-3-sonnet",
)

# First message
result1 = await agent("My name is John").collect()

# Second message - agent remembers John's name
result2 = await agent("What's my name?").collect()

# Agent responds: "Your name is John"`}/>

## Automatic Memory Resolution

Complex workflows require maintaining state across multiple steps and agent interactions. -> Workflow state management with automatic context passing

### Example 2: Multi-Step Workflow Context

<CodeBlock language="python" code ={`from timbal.core import Agent

# Product finder agent
product_finder = Agent(
    name="product_finder",
    model="openai/gpt-4o-mini",
    system_prompt="Find 5 laptops with: brand, model, price, and specs. Keep under 100 words."
)

# Price optimizer that selects best deal
price_optimizer = Agent(
    name="price_optimizer",
    model="openai/gpt-4o-mini",
    tools=[product_finder],
    system_prompt="Get laptops using product_finder tool, then recommend the best value option."
)

result = await price_optimizer(
    prompt=Message.validate("Find me the best laptop under $1000")
).collect()`}/>

When the parent calls the child agent, memory is automatically shared. The child agent automatically receives the full conversation context including the original user request and any previous exchanges.

The `price_optimizer` automatically knows the laptop list from `product_finder` through conversation context.

#### How Automatic Memory Works

When a child agent is called as a tool:

1. **Context Retrieval**: The child agent automatically retrieves the parent's conversation history
2. **Memory Reconstruction**: All previous messages (user inputs, assistant responses, tool calls) are included
3. **Seamless Continuation**: The child agent responds with full context awareness

### Example 3: User Preference Management

Different users have different preferences for AI behavior, language, and detail level. -> Context-aware default parameters with user preferences

<CodeBlock language="python" code ={`from timbal.core import Agent
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


## Explicit State Management

For persistent memory across separate sessions, use RunContext:

<CodeBlock language="python" code ={`from timbal.core import Agent
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

### Exemple 4: Long Conversation Management

Very long conversations exceed token limits and need intelligent summarization. -> Conversation summarization and context compression

<CodeBlock language="python" code ={`from timbal.core import Agent
from timbal.state import get_run_context

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

### Example 5: Multi-User Session Management

Managing multiple concurrent user sessions with different contexts -> Session-based context management

<CodeBlock language="python" code ={`from timbal.core import Agent
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

## Summary

Timbal's memory management provides:

- **Automatic Memory Resolution**: Seamless context sharing for nested agents
- **Enhanced State Management**: Improved RunContext and state saver integration  
- **Flexible Memory Patterns**: Custom memory resolution and filtering
- **Performance Optimization**: Token-aware memory compression
- **Rich Debugging**: Comprehensive tracing and logging

The architecture makes memory handling more robust and automatic while providing advanced customization options for complex use cases.