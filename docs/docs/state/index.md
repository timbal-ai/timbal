---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Understanding Memories

Memories in Timbal enable AI systems to maintain context and track conversations across interactions. This guide explains how to implement and configure memory systems for your AI applications.

## What are Memories?

Memories in Timbal provide persistent context management that:
- Tracks conversation history
- Maintains context between interactions
- Enables natural conversation flow
- Supports personalized AI experiences

## How Memories Work

The memory system consists of two core components:

1. **Message History**
   - Stores conversation history
   - Tracks tool usage and results
   - Maintains conversation flow
   - Preserves important context

2. **State Management**
   - Saves memory between sessions
   - Manages memory window size
   - Handles memory persistence
   - Provides different storage options

## Setting Up Memories

To implement memory in your AI system you can do it as in Flow or Agents:

<CodeBlock language="python" code ={`from timbal import Flow
from timbal.state.savers import InMemorySaver

# Create a flow with memory
flow = (
      Flow()
      .add_llm(
          id="chat_llm",
          memory_id="chat_llm",  # Enable memory
          memory_window_size=30   # Remember last 30 messages
    )
)

# Add state management
flow.compile(state_saver=InMemorySaver())`}/>


<CodeBlock language="python" code ={`from timbal import Agent
from timbal.state.savers import InMemorySaver

# Create a flow with memory
Agent(
        model="gpt-4o-mini",
        state_saver=InMemorySaver(),
    )`}/>


## Memory Configuration Options

Configure memory according to your needs:

1. **Shared Memory**
<CodeBlock language="python" code ={`# Both LLMs share the same memory
   flow = (
         Flow()
         .add_llm(id="llm1", memory_id="shared_memory")
         .add_llm(id="llm2", memory_id="shared_memory")
   )`}/>

2. **Limited Memory Window**
<CodeBlock language="python" code ={`flow.add_llm(
          id="llm",
          memory_id="llm",
          memory_window_size=10  # Remember only last 10 messages
   )`}/>

## State Savers

Choose an appropriate state saver for your use case:

1. **InMemorySaver**
   <div style={{display: 'flex', alignItems: 'flex-start', gap: '-4rem', margin: '0em 0'}}>
     <div style={{flex: 1}}>
       <ul>
         <li>Development and testing</li>
         <li>RAM-based storage</li>
         <li>Clears on program termination</li>
       </ul>
     </div>
     <div style={{flex: 1}}>
       <CodeBlock language="python" code={`from timbal.state.savers import InMemorySaver
   state_saver=InMemorySaver()`}/>
     </div>
   </div>

2. **JSONLSaver**
   <div style={{display: 'flex', alignItems: 'flex-start', gap: '-4rem', margin: '0em 0'}}>
     <div style={{flex: 1}}>
       <ul>
         <li>File-based storage</li>
         <li>Simple applications</li>
         <li>Persists between restarts</li>
       </ul>
     </div>
     <div style={{flex: 1}}>
       <CodeBlock language="python" code={`from timbal.state.savers import JSONLSaver
   state_saver=JSONLSaver("memories.jsonl")`}/>
     </div>
   </div>

3. **TimbalPlatformSaver**
   <div style={{display: 'flex', alignItems: 'flex-start', gap: '-4rem', margin: '0em 0'}}>
      <div style={{flex: 1}}>
         <ul>
            <li>Production environment</li>
            <li>Managed by Timbal Platform</li>
            <li>Scalable and reliable</li>
         </ul>
      </div>
      <div style={{flex: 1}}>
         <CodeBlock language="python" code={`from timbal.state.savers import TimbalPlatformSaver
   state_saver=TimbalPlatformSaver()`}/>
      </div>
      </div>

## Memory Features

The memory system provides:

- **Context Preservation**
  - Maintains conversation flow
  - Tracks user preferences
  - Records previous decisions

- **Window Management**
  - Controls memory size
  - Preserves important context
  - Optimizes performance

- **Tool Context**
  - Records tool usage results
  - Maintains tool call context
  - Links related interactions