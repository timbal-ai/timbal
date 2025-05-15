---
title: Adding Memory
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Using Memory in Agents

Memory in Timbal is implemented through state savers, which allow agents to maintain context across multiple interactions. This is achieved through a combination of <span style={{color: 'var(--timbal-purple)'}}><strong>state savers</strong></span> and <span style={{color: 'var(--timbal-purple)'}}><strong>run contexts</strong></span>.

---

## State Savers

State savers are responsible for persisting the agent's state across multiple interactions. They store snapshots of the agent's state, including:
- Input/output messages
- Memory content
- Run metadata (timestamps, usage statistics)
- Error information (if any)

The most basic state saver is the `InMemorySaver`, which stores everything in memory. Here's how to use it:

<CodeBlock language="python" highlight="7" code ={`from timbal import Agent
from timbal.state.savers import InMemorySaver

# Initialize agent with memory
agent = Agent(
    model="gpt-4o-mini",
    state_saver=InMemorySaver(),
)`}/>

## Run Context and Memory Chain

Memory is maintained through a chain of run contexts, where each interaction is linked to its previous one through a `parent_id`. This creates a traceable history of conversations.

Here's a complete example showing how memory works:

<CodeBlock language="python" highlight="8,12,14,19,21" code ={`from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import InMemorySaver

# Initialize agent with memory
agent = Agent(
    model="gpt-4o-mini",
    state_saver=InMemorySaver(),
)

# First interaction - no parent context
run_context = RunContext()
flow_output_event = await agent.complete(
    context=run_context,
    prompt="My name is David"
)

# Second interaction - using parent context
run_context = RunContext(parent_id=flow_output_event.run_id)
flow_output_event = await agent.complete(
    context=run_context,
    prompt="What's my name?"
)`}/>

The agent responded with _"Your name is David"_ because it has access to the previous context.

Let's take a closer look at the registers:

<div className="log-step-static">
StartEvent(..., path='agent', ...)
</div>
<div className="log-step-static">
StartEvent(..., path='agent.llm-0', ...)
</div>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='agent.llm-0', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(..., 
    path='agent.llm-0',
    input={
        'messages': [Message(
                role=user,
                content=[TextContent(
                    type='text',
                    text='My name is David'
                )]
        )], 
        'model': 'gpt-4o-mini',
        ...
    },
    output=Message(
        role=assistant, 
        content=[TextContent(
            type='text', 
            text='Nice to meet you, David! How can I assist you today?'
        )]
    )
    ...
)`}/>
</details>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='agent', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='agent', 
    input={
        'prompt': {
            'role': 'user', 
            'content': [{'type': 'text', 'text': 'My name is David'}]
        }
    },
    output=Message(
        role=assistant, 
        content=[TextContent(
            type='text', 
            text='Nice to meet you, David! How can I assist you today?'
        )]
    ), 
    ...
)`}/>
</details>
<div className="log-step-static">
StartEvent(..., path='agent', ...')
</div>
<div className="log-step-static">
StartEvent(..., path='agent.llm-0', ...')
</div>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='agent.llm-0', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(..., 
    path='agent.llm-0',
    input={
        'messages': [
            Message(
                role=user, 
                content=[TextContent(type='text', text='My name is David')]
            ),
            Message(
                role=assistant, 
                content=[TextContent(
                    type='text', 
                    text='Nice to meet you, David! How can I assist you today?'
                )]
            ),
            Message(
                role=user,
                content=[TextContent(type='text', text="What's my name?")]
            )
        ],
        'model': 'gpt-4o-mini', 
        ...
    },
    output=Message(
        role=assistant, 
        content=[TextContent(type='text', text='Your name is David! How can I help you today?')]
    ), ...
)`}/>
</details>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='agent', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(...,
    path='agent',
    input={
        'prompt': {
            'role': 'user', 
            'content': [{'type': 'text', 'text': "What's my name?"}]
        }
    }, 
    output=Message(
        role=assistant, 
        content=[TextContent(type='text', text='Your name is David! How can I help you today?')]
    ),
    ...
)`}/>
</details>

## How Memory Works

1. **Run ID**: Each interaction gets a unique `run_id` that identifies that specific interaction.

2. **Parent ID**: When you want to maintain context, you create a new `RunContext` with the `parent_id` set to the previous interaction's `run_id`. This creates a chain of memory.

3. **Memory Loading**: When an agent receives a request with a `parent_id`, it:
   - Loads the previous state from the state saver
   - Includes that context in the current interaction
   - Maintains the conversation history

### Memory Chain Visualization

<CodeBlock language="bash" highlight="2,4,6" code ={`
Interaction 1 (run_id: abc123)
└── "My name is David and I live in Barcelona"
    └── Interaction 2 (run_id: def456, parent_id: abc123)
        └── "Where do I live?"
            └── Interaction 3 (run_id: ghi789, parent_id: def456)
                └── "And what's my name?"`}/>

Each interaction maintains its connection to the previous one through the `parent_id`, creating a chain of memory that allows the agent to maintain context throughout the conversation.

For more information about different types of state savers and their implementations, check out the [State documentation](/state).