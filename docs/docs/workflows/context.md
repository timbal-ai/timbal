---
title: Context 
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Context

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Share data between Workflow steps.
</h2>

---


## Step Variables

Every step in a workflow has access to two built-in variables:

- **`.input`**: Contains all the parameters passed to the step. They are accessed through their name.
- **`.output`**: Contains the value(s) returned by the step. Can be a single value, dictionary, array, or custom class.

Additionally, **custom variables** can be linked to a step using the [`get_run_context().set_data`](#the-get_run_context-function) method.

<CodeBlock language="python" highlight="6" code ={`import asyncio
import datetime
from timbal import Workflow, get_run_context

async def process_user_data(user_id: str):
    get_run_context().set_data(".user_status", "active")
    return f"Processed user: {user_id}"

async def send_notification(message: str, user: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [message, user, timestamp]


workflow = (Workflow(name='user_workflow')
    .step(process_user_data, user_id="user_123")
    .step(send_notification, message="Welcome!", user="user_123")
)`}/>

After the workflow runs, each step has the following variables:

#### `process_user_data`:
<CodeBlock language="python" code={`.input.user_id       = 'user_123'
.output              = 'Processed user: user_123'
.user_status         = 'active'`}/>

#### `send_notification`:
<CodeBlock language="python" code={`.input.message       = 'Welcome!'
.input.user          = 'user_123'
.output[0]           = 'Welcome!'
.output[1]           = 'user_123'
.output[2]           = '2025-01-15 14:30:25'`}/>


## Accessing the Context
### The `get_run_context()` Method
This function provides access to the current run context and allows you to read and write data that can be shared across workflow steps.

- **`get_data(path)`**:
  - Returns the value from the specified path.
  - Example: `get_run_context().get_data(".input.user_id")`

- **`set_data(path, value)`**:
  - Creates or updates the value at the given path.
  - Data persists and can be accessed by other steps.
  - Example: `get_run_context().set_data(".user_status", "active")`


### Variable Access
Context paths use a hierarchical structure, enabling each step to access both its own variables and those from other steps within the workflow.

- **Current step**: Use `.` to access variables from the current step
  - `.input.parameter_name`
  - `.output`
  - `.custom_variable`

- **Parent step**: Use `..` to access data from the parent run in the workflow
  - `..output`
  - `..custom_variable`

- **Neighbour steps**: Use the step name to access data from any neighbour step
  - `step_name.output`
  - `step_name.custom_variable`



<!-- To access context variables within your step functions, use the `get_run_context()` function: -->

<CodeBlock language="python" highlight="2" code={`async def check_status():
    status = get_run_context().get_data("process_user_data.user_status")
    print(f"Your current status is: {status}")

workflow = (Workflow(name='user_workflow')
    .step(process_user_data, user_id="user_123")
    .step(send_notification, message="Welcome!", user="user_123")
    .step(check_status)
)`}/>

