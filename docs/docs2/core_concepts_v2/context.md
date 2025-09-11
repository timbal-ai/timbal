---
title: Context & State Management
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Context & State Management

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Centralized runtime execution context, that serves as the shared state management system across all components in a run.
</h2>

---
## The RunContext

The <span style={{color: 'var(--timbal-purple)'}}><strong>RunContext</strong></span> is the central state management and execution coordination system in Timbal that serves as the shared memory and communication hub for all components within a single and complete execution of a Runnable. It acts as a centralized container that maintains execution state, enables data sharing between components, tracks usage metrics, and manages the hierarchical relationships between parent and child runs.

### Accessing the RunContext
The function `get_run_context()` provides access to the current execution's shared state container. You can interact with it using two main methods: `get_data(key)` to retrieve values from the shared state or `set_data(key, value)` to store or update data. The context is automatically available within any Runnable execution (functions, tools, agents, or workflows)


<CodeBlock language="python" code={`from timbal.state import get_run_context

async def process_user_data(user_id: str):
    # Access the current RunContext
    context = get_run_context()
    
    # Store custom data in the context
    context.set_data(".user_status", "active")
    context.set_data(".processed_at", "2025-01-15")
    
    return f"Processed user: {user_id}"`}/>

<!-- It includes comprehensive tracing capabilities that store detailed execution information (input, output, error, timing, and usage data), platform configuration management, and sophisticated data access methods that allow components to reference data from different parts of the execution hierarchy using a flexible key format. -->

---


## Data Sharing Notation
Data sharing capabilities are provided through a hierarchical key format that enables components to access data across different levels of the execution hierarchy. There are three main access patterns:

- **Current Runnable Access** (`.` prefix): Access to their own data.
    - `.input.parameter_name`
    - `.output`
    - `.custom_variable_name`

- **Parent Runnable Access** (`..` prefix): Access data from their immediate parent in the execution hierarchy. For example, the `..input` retrieves the input from the parent Runnable, enabling child components to build upon parent results.

- **Neighbours Runnable Access**: Access data from neighbour Runnables. See section [Workflow Context](../workflows_v2/context.md) for more details.