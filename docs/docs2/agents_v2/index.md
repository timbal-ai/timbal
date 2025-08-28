---
title: Overview
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Understanding Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master proven strategies for designing advanced, specialized AI agents using an architecture that work together seamlessly to tackle complex challenges.
</h2>

---

## What are Agents?

<span style={{color: 'var(--timbal-purple)'}}><strong>Agents</strong></span> are autonomous execution units that **orchestrate LLM interactions with tool calling**. 

Without tools, an agent functions as a basic LLM. The simplest agent requires just a name and model:

<CodeBlock language="python" code ={`from timbal import Agent

agent = Agent(
    name="my_agent", # required
    model="openai/gpt-5"
)  # That's it! You've created your first agent!`}/>

You can specify any model using the "provider/model" format. See all supported models in [Model Capabilities](/getting-started/model_capabilities).

**Note:** Make sure to define all required environment variables—such as the API key model that you need—in your `.env` file.

<CodeBlock language="bash" title=".env" code ={`OPENAI_API_KEY=your_api_key_here`}/>

Define tools as Python functions - the framework handles schema generation, parameter validation, and execution orchestration. The framework performs automatic introspection of function signatures and docstrings for tool schema generation.

## Quick Example

Here's an example of an Agent that solves a real business problem - customer support automation:

<CodeBlock language="python" code ={`from timbal import Agent
from timbal.types.message import Message

# Define tools for customer support
def search_knowledge_base(query: str) -> str:
    """Search company knowledge base for relevant information."""
    # Implementation would connect to your knowledge base
    return f"Found information about: {query}"

def create_support_ticket(issue: str, user_email: str) -> str:
    """Create a support ticket for complex issues."""
    # Implementation would integrate with your ticketing system
    return f"Created ticket for {user_email}: {issue}"

def escalate_to_human(user_email: str, reason: str) -> str:
    """Escalate complex issues to human agents."""
    return f"Escalated {user_email} to human agent: {reason}"

# Create a customer support agent
agent = Agent(
    name="customer_support_agent",
    model="anthropic/claude-3-sonnet",
    tools=[search_knowledge_base, create_support_ticket, escalate_to_human],
    system_prompt="""You are a customer support agent.
    Your goal is to help customers efficiently:
    1. First, try to answer their question using the knowledge base
    2. If you can't find an answer, create a support ticket
    3. For complex technical issues, escalate to a human agent
    4. Always be polite and professional"""
)

# Use the agent to handle a customer request
response = await agent(
    "I can't log into my account and I've tried resetting my password"
  ).collect()`}/>

**What happens behind the scenes:**

<div className="log-step-static">
  StartEvent(..., path='customer_support_agent, ...)
</div>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='customer_support_agent.llm-0', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='customer_support_agent.llm-0', 
    input={
      'messages': [
        Message(
          role=user,
          content=[TextContent(
            type='text', 
            text="I can't log into my account and I've tried resetting my password"
          )]
        )
      ], 
      'tools': [{
        'type': 'function', 
        'function': {
          'name': 'search_knowledge_base',
          'description': 'Search company knowledge base for relevant information',
          'parameters': {
            'properties': {'query': {'title': 'Query', 'type': 'string'}}, 
            'required': ['query'],
            ...
          }
        }
      }], 
      'model': 'claude-3-sonnet',
      ...
    },
    output=Message(
      role=assistant,
      content=[ToolUseContent(
        type='tool_use', 
        id='...', 
        name='search_knowledge_base', 
        input={'query': 'login password reset troubleshooting'}
      )]
    ), ...)
`}/>
</details>

<div className="log-step-static">
  StartEvent(..., path='customer_support_agent.search_knowledge_base-call_...', ...)
</div>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='customer_support_agent.search_knowledge_base-...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='customer_support_agent.search_knowledge_base-...',
    input={'query': 'login password reset troubleshooting'},
    output=Message(
      role=user,
      content=[TextContent(type='text', text='Found information about: login password reset troubleshooting')]
    ), ...)`}/>
</details>

<div className="log-step-static">
  StartEvent(..., path='customer_support_agent.llm-1', ...)
</div>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='customer_support_agent.llm-1', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='customer_support_agent.llm-1', 
    input={
      'messages': [
        Message(
          role=user, 
          content=[TextContent(
            type='text',
            text="I can't log into my account and I've tried resetting my password"
          )]
        ), 
        Message(
          role=assistant, 
          content=[ToolUseContent(
            type='tool_use',
            id='...',
            name='search_knowledge_base',
            input={'query': 'login password reset troubleshooting'}
          )]
        ),
        Message(
          role=user, 
          content=[ToolResultContent(
            type='tool_result', 
            id='call_...', 
            content=[TextContent(type='text', text='Found information about: login password reset troubleshooting')]
          )]
        )
      ],
      'tools': [{
        'type': 'function', 
        'function': {
          'name': 'create_support_ticket',
          'description': 'Create a support ticket for complex issues',
          'parameters': {
            'properties': {
              'issue': {'title': 'Issue', 'type': 'string'},
              'user_email': {'title': 'User Email', 'type': 'string'}
            },
            'required': ['issue', 'user_email'],
            ...
          }
        }
      }], 
      'model': 'claude-3-sonnet',
      ...
    },
    output=Message(
      role=assistant, 
      content=[ToolUseContent(
        type='tool_use',
        id='...',
        name='create_support_ticket',
        input={
          'issue': 'User cannot log in despite password reset attempts',
          'user_email': 'user@example.com'
        }
      )]
    ),...)`}/>
</details>

<details className="log-step-collapsible">
<summary>
  OutputEvent(..., path='customer_support_agent', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='customer_support_agent',
    input={
      'prompt': {
        'role': 'user', 
        'content': [{
          'type': 'text',
          'text': "I can't log into my account and I've tried resetting my password"
        }]
      }
    },
    output=Message(
      role=assistant, 
      content=[TextContent(
        type='text',
        text='I understand you\'re having trouble logging in even after trying to reset your password. This sounds like a more complex issue that requires technical investigation. I\'ve created a support ticket for you, and our technical team will investigate this issue. You should receive an email confirmation shortly with your ticket number.'
      )]
    ), ...)
`}/>
</details>

<div style={{marginTop: '2rem'}}>
This example shows how an agent can autonomously handle complex customer support scenarios by reasoning about the problem, searching for solutions, and taking appropriate actions.
</div>

## Architecture features


- <span style={{color: 'var(--timbal-purple)'}}><strong>Execution Engine</strong></span>:
    - Asynchronous concurrent tool execution via multiplexed event queues
    - Conversation state management with automatic memory persistence across iterations
    - Multi-provider LLM routing with unified interface abstraction
  
- <span style={{color: 'var(--timbal-purple)'}}><strong>Tool System</strong></span>:
  - Runtime tool discovery with automatic OpenAI/Anthropic schema generation
  - Support for nested Runnable composition and hierarchical agent orchestration
  - Dynamic parameter validation using Pydantic models

- <span style={{color: 'var(--timbal-purple)'}}><strong>Advanced Runtime</strong></span>:
  - Template-based system prompt composition with runtime callable injection
  - Configurable iteration limits with autonomous termination detection
  - Event-driven streaming architecture with real-time processing capabilities
  - Pre/post execution hooks for cross-cutting concerns and runtime interception


## Key Features

### Autonomous Execution Loop

Agent implements a sophisticated autonomous execution pattern:

1. **Receive Input**: Accept a user promp
2. **Memory Resolution**: Load conversation history from parent context if nested
3. **LLM Interaction**: Call the LLM with available tools and conversation history
4. **Tool Execution**: Execute tool calls concurrently for better performance
5. **Iteration**: Continue until no more tool calls or max_iter is reached

### Enhanced Tool Management

<CodeBlock language="python" code ={`# Tools can be defined in multiple ways
agent = Agent(
    name="multi_tool_agent",
    model="openai/gpt-4",
    tools=[
        # 1. Direct function
        search_web,
        
        # 2. Tool instance with custom config
        Tool(
            handler=query_database,
            description="Query the customer database",
            exclude_params=["connection_string"]
        ),
        
        # 3. Dictionary configuration
        {
            "handler": send_notification,
            "description": "Send notification to user",
            "params_mode": "required"
        }
    ]
)`}/>

Learn more about creating and using tools in the [Tools guide](/agents/tools).

### System Prompt Configuration

Agent uses a `system_prompt` parameter to provide context and instructions:

<CodeBlock language="python" code ={`agent = Agent(
    name="specialized_agent",
    model="anthropic/claude-3-sonnet",
    system_prompt="""You are a data analyst specialized in financial data. 
    Your role is to:
    1. Analyze financial data accurately
    2. Provide clear explanations of trends
    3. Identify potential risks and opportunities
    4. Present findings in a professional manner""",
    tools=[analyze_financial_data, generate_charts, create_reports]
)

# Use the agent with the configured system prompt
result = await agent("Analyze this quarterly financial data").collect()`}/>

## Running an Agent

Agent provides a streamlined execution interface:

### Get a Complete Answer

<CodeBlock language="python" code ={`result = await agent(
    prompt="What's the current market trend for renewable energy stocks?"
).collect()`}/>

### Real-Time (Streaming) Output

<CodeBlock language="python" code ={`async for event in agent(
    prompt="What's the current market trend for renewable energy stocks?"
):
    print(event)`}/>

Here you can see the actual output from running this example:

<div className="log-step-static">
  type='START' run_id='068a74bf-8b16-7de3-8000-614f2c07a86e' path='specialized_agent' status_text=None
</div>

<details className="log-step-collapsible">
<summary>
  type='OUTPUT' run_id='068a74bf-8b16-7de3-8000-614f2c07a86e' path='specialized_agent'
</summary>
<CodeBlock language="bash" code={`type='OUTPUT' run_id='068a74bf-8b16-7de3-8000-614f2c07a86e' path='specialized_agent' 
input={'prompt': 'What\'s the current market trend for renewable energy stocks?'} 
output=Message(role=assistant, content=[TextContent(type='text', text="I don't have access to real-time market data, but I can provide general insights about renewable energy stock trends based on recent developments in the sector.")]) 
error=None 
t0=1755794424693 t1=1755794425981 
usage={'claude-3-sonnet:input_text_tokens': 45, 'claude-3-sonnet:output_text_tokens': 35}`}/>
</details>

## Parameter Validation

Agent uses Pydantic models for robust parameter validation:

<CodeBlock language="python" code ={`from timbal.types.message import Message

# Input is automatically validated
result = await agent(
    prompt=Message.validate("Hello"),  # Must be a Message
).collect()`}/>

## Next Steps

- Try creating your own Agent with different tools
- Experiment with system prompt templates
- Explore concurrent tool execution patterns
- See examples in [Examples](/examples)

Remember: The more you practice, the better you'll become at creating powerful Agents!
