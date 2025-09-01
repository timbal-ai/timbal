---
title: Running Workflows
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows can be run from different environments. These examples demonstrate how to execute a workflow using a command line script or by calling the workflow directly from a client-side component.

## From the command line

In this example, a run script has been added to the `src` directory. The workflow processes input data and executes sequential steps.

<CodeBlock language="python" title="test_run_workflow.py" code={`import asyncio
from timbal.core import Workflow

# Define workflow steps
def step_1(value: int) -> dict:
    """First step that processes the input value."""
    return {"value": value, "step": 1}

def step_2(value: int) -> dict:
    """Second step that processes the input value."""
    return {"value": value, "step": 2}

# Create the workflow
sequential_workflow = (
    Workflow(name="sequentialSteps")
    .step(step_1)
    .step(step_2)
    .link("step_1", "step_2")
)

async def main():
    # Run the workflow with input data
    result = await sequential_workflow(value=10).collect()
    
    print("Workflow execution completed!")
    print(f"Final result: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())`}/>

### Run the script

Run the workflow using the following command:

<CodeBlock language="bash" code={`python test_run_workflow.py`}

Or if you prefer using `uv`:

<CodeBlock language="bash" code={`uv run test_run_workflow.py`}

### Command line output

The output from this workflow run will look similar to the below:


<CodeBlock language="bash" code={`{
 start_event                    call_id=068b5bc01c127ae88000ebd721ce0e98 parent_call_id=None parent_run_id=None path=sequentialSteps run_id=068b5bc0-1c12-7be4-8000-87f69e630703 status_text=None type=START
2025-09-01 17:30:09 [info     ] start_event                    call_id=068b5bc01c1578428000b215329a9592 parent_call_id=068b5bc01c127ae88000ebd721ce0e98 parent_run_id=None path=sequentialSteps.step_1 run_id=068b5bc0-1c12-7be4-8000-87f69e630703 status_text=None type=START
2025-09-01 17:30:09 [info     ] output_event                   call_id=068b5bc01c1578428000b215329a9592 error=None input={'value': 10} output={'value': 10, 'step': 1} parent_call_id=068b5bc01c127ae88000ebd721ce0e98 parent_run_id=None path=sequentialSteps.step_1 run_id=068b5bc0-1c12-7be4-8000-87f69e630703 t0=1756740609755 t1=1756740609756 type=OUTPUT usage={}
2025-09-01 17:30:09 [info     ] start_event                    call_id=068b5bc01c1a71768000be84dab7a431 parent_call_id=068b5bc01c127ae88000ebd721ce0e98 parent_run_id=None path=sequentialSteps.step_2 run_id=068b5bc0-1c12-7be4-8000-87f69e630703 status_text=None type=START
2025-09-01 17:30:09 [info     ] output_event                   call_id=068b5bc01c1a71768000be84dab7a431 error=None input={'value': 10} output={'value': 10, 'step': 2} parent_call_id=068b5bc01c127ae88000ebd721ce0e98 parent_run_id=None path=sequentialSteps.step_2 run_id=068b5bc0-1c12-7be4-8000-87f69e630703 t0=1756740609756 t1=1756740609756 type=OUTPUT usage={}
2025-09-01 17:30:09 [info     ] output_event                   call_id=068b5bc01c127ae88000ebd721ce0e98 error=None input={'value': 10} output={'value': 10, 'step': 2} parent_call_id=None parent_run_id=None path=sequentialSteps run_id=068b5bc0-1c12-7be4-8000-87f69e630703 t0=1756740609754 t1=1756740609756 type=OUTPUT usage={}
}`}


## Alternative: Simple workflow execution

For simpler use cases, you can run workflows directly without complex setup:

<CodeBlock language="python" code={`# Simple workflow execution
from timbal.core import Workflow

# Define a simple workflow
simple_workflow = (
    Workflow(name="simple")
    .step(lambda x: x * 2)
    .step(lambda x: x + 1)
)

async def run_simple():
    # Execute with input value 5
    result = await simple_workflow(x=5).collect()
    print(f"Input: 5, Output: {result.output}")

# Run the simple workflow
asyncio.run(run_simple())
`}/>