---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Understanding Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Master proven strategies for designing advanced, specialized AI agents using an architecture that work together seamlessly to tackle complex challenges.
</h2>

---

## What is an Agent?

An <span style={{color: 'var(--timbal-purple)'}}><strong>Agent</strong></span> is an AI system that can autonomously reason, make decisions, and take actions to achieve specific goals. Unlike simple chatbots, agents can:

- **Analyze complex problems** and break them down into manageable steps
- **Choose appropriate tools** from their available toolkit based on the task
- **Execute multiple actions** in sequence or parallel to solve problems
- **Learn from interactions** and adapt their approach over time
- **Maintain context** across multiple conversations and sessions

<CodeBlock language="python" code ={`from timbal import Agent

agent = Agent(
    name="my_agent",
    model="anthropic/claude-3-sonnet"
)  # That's it! You've created your first agent!`}/>

The `name` parameter is required and provides a unique identifier for your agent. The `model` parameter specifies the provider and model to use for the agent.

**Note:** Make sure to define all required environment variables—such as the API key model that you need—in your `.env` file.

<CodeBlock language="bash" title=".env" code ={`ANTHROPIC_API_KEY=your_api_key_here`}/>

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
    system_prompt="""You are a customer support agent. Your goal is to help customers efficiently:
    1. First, try to answer their question using the knowledge base
    2. If you can't find an answer, create a support ticket
    3. For complex technical issues, escalate to a human agent
    4. Always be polite and professional"""
)

# Use the agent to handle a customer request
response = await agent("I can't log into my account and I've tried resetting my password").collect()`}/>

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

## Key Capabilities of an Agent

Let's break down how an Agent thinks and works:

<div className="timeline">
<div className="timeline-item">
<div className="timeline-content">

<h4>Autonomous Reasoning</h4>
Agents can analyze complex problems, break them down into steps, and decide the best approach to solve them.

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Tool Selection</h4>
They can choose the most appropriate tools from their available toolkit based on the current situation and requirements.

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Multi-Step Execution</h4>
Agents can execute multiple actions in sequence, making decisions based on the results of previous steps.

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Context Awareness</h4>
They maintain awareness of the current situation, user preferences, and conversation history to provide relevant responses.

</div>
</div>
</div>

<style>{`
.timeline {
  display: flex;
  align-items: center;
  margin: 1rem 0;
  overflow-x: auto;
  padding: 0.5rem;
}

.timeline-item {
  width: 180px;
  text-align: left;
  padding: 0.5rem;
  background: var(--ifm-background-color);
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.timeline-connector {
  color: var(--ifm-color-primary);
  font-size: 1.2rem;
  padding: 0 0.5rem;
}

.timeline-content {
  padding: 0.5rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.timeline-content h3 {
  margin-bottom: 0.25rem;
  font-size: 1rem;
}

.timeline-content ul {
  list-style: disc;
  padding-left: 1.2em;
  margin: 0;
}

.timeline-content li {
  margin: 0.15rem 0;
  font-size: 0.85rem;
}

.timeline-content h4 {
  color: var(--ifm-color-primary);
  font-weight: bold;
  margin-bottom: 0.5em;
  margin-top: 0;
}
`}</style>

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
result = await agent(
    prompt=Message(role="user", content="Analyze this quarterly financial data")
).collect()`}/>

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
