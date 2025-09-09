---
title: Control Flow
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Control Flow

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Workflows are broken into multiple steps, which can be executed following different behaviours.
</h2>

---

## Default Execution
Steps run in parallel by default. If one step produces an output needed by another, **the framework automatically enforces sequential execution between those steps**. The dependency is resolved under the hood.

<!-- TODO: Add code example of dependencies -->

<CodeBlock language="python" code ={`import asyncio
from timbal import Workflow

async def func_1(x: str):
    asyncio.sleep(1)
    return x

async def func_2(y: str):
    asyncio.sleep(10)
    return y

async def func_3():
    return get_run_context().get_data("func_1.output") # output from step func_1

workflow = (Workflow(name='my_workflow')
    .step(func_1, x="a")
    .step(func_2, y="b")
    .step(func_3)
)`}/>

In this example, we have:
- `func_1` and `func_2` run concurrently.
- `func_3` starts only after `func_1` has finished.
- The workflow's final output will be `b`, since `func_2` it is the last to finish.


## Forcing Sequential Execution:
Steps can be run sequentially even if there is no data dependency. The `depends_on` parameter allows to explicitly control ordering:

<CodeBlock language="python" code ={`workflow = (Workflow(name='my_workflow')
    .step(func_1, x="a")
    .step(func_2, y="b", depends_on=[func_1])
    .step(func_3)
)`}/>

Now, both steps `func_2`and `func_3` must wait for `func_1` to finish.


## Conditional Execution

The `when` ensures that a step is executed only if the specified condition is met.


<CodeBlock language="python" code ={`async def send_alert(message: str):
    print(f"Alert: {message}")


workflow = (Workflow(name='my_workflow')
    .step(func_1, x="a")
    .step(func_2, y="b")
    .step(func_3)
    .step(send_alert, message="Process completed",
        when=lambda: get_run_context().get_data("func_1.output") and 
            get_run_context().get_data("func_3.output")) 
)`}/>

In this case, `send_alert` step will be executed after both `func_1` and `func_2` have been executed and have an output. 


## Summary
- Parallel execution by default.
- Sequential execution is applied automatically when parameter dependencies exist between steps.
- Sequential execution can be enforced with `depends_on`.
- Conditional execution of steps is specified with `when`.
