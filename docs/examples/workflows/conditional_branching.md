---
title: Conditional Branching
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows often need to follow different paths based on a condition. These examples demonstrate how to create conditional flows using Timbal's workflow orchestration capabilities.

## Conditional logic using steps

In this example, the workflow uses conditional logic to execute one of two steps based on a condition. If the input value is less than or equal to 10, it runs `less_than_step` and returns 0. If the value is greater than 10, it runs `greater_than_step` and returns 20.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool

# Define the less than step
def less_than_step(value: int) -> dict:
    """If value is <= 10, return 0."""
    return {"value": 0}

# Define the greater than step
def greater_than_step(value: int) -> dict:
    """If value is > 10, return 20."""
    return {"value": 20}

# Define the conditional logic step
def conditional_branch(value: int) -> dict:
    """Determines which step to execute based on the input value."""
    if value <= 10:
        return less_than_step(value)
    else:
        return greater_than_step(value)

# Create the conditional workflow
branch_workflow = (
    Workflow(name="branch-workflow")
    .step(conditional_branch)
)

# Alternative: Using Tool class for more complex conditional logic
def conditional_branch_tool(value: int) -> dict:
    """Tool that handles conditional branching logic."""
    if value <= 10:
        return {"value": 0, "branch": "less_than"}
    else:
        return {"value": 20, "branch": "greater_than"}

conditional_tool = Tool(
    name="conditional_branch",
    description="Executes different logic based on input value",
    handler=conditional_branch_tool
)

tool_branch_workflow = (
    Workflow(name="tool-branch-workflow")
    .step(conditional_tool)
)
`}/>

## Conditional logic using workflows

In this example, the workflow uses conditional logic to execute one of two nested workflows based on a condition. If the input value is less than or equal to 10, it runs `less_than_workflow`. If the value is greater than 10, it runs `greater_than_workflow`.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool

# Define individual workflow steps
def less_than_step(value: int) -> dict:
    """If value is <= 10, return 0."""
    return {"value": 0}

def greater_than_step(value: int) -> dict:
    """If value is > 10, return 20."""
    return {"value": 20}

# Create individual workflows
less_than_workflow = (
    Workflow(name="less-than-workflow")
    .step(less_than_step)
)

greater_than_workflow = (
    Workflow(name="greater-than-workflow")
    .step(greater_than_step)
)

# Define the main conditional workflow
def main_conditional_workflow(value: int) -> dict:
    """Main workflow that chooses which sub-workflow to execute."""
    if value <= 10:
        # Execute less than workflow
        return less_than_workflow(value=value).collect()
    else:
        # Execute greater than workflow
        return greater_than_workflow(value=value).collect()

# Create the main conditional workflow
branch_workflows = (
    Workflow(name="branch-workflows")
    .step(main_conditional_workflow)
)

# Alternative: Using async conditional execution
async def async_conditional_workflow(value: int) -> dict:
    """Async version that can handle async sub-workflows."""
    if value <= 10:
        result = await less_than_workflow(value=value).collect()
        return result.output
    else:
        result = await greater_than_workflow(value=value).collect()
        return result.output

async_branch_workflows = (
    Workflow(name="async-branch-workflows")
    .step(async_conditional_workflow)
)
`}/>

## How conditional branching works in Timbal

1. **Conditional Logic**: Use Python's native `if/else` statements for branching
2. **Step Selection**: Choose which step or workflow to execute based on conditions
3. **Workflow Composition**: Combine multiple workflows with conditional execution
4. **Async Support**: Handle both synchronous and asynchronous conditional logic
5. **Flexible Outputs**: Each branch can return different data structures

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Test with different input values
    
    # Test value <= 10
    print("=== Testing with value 5 (<= 10) ===")
    result1 = await branch_workflow(value=5).collect()
    print(f"Input: 5, Output: {result1.output}")
    
    # Test value > 10
    print("\n=== Testing with value 15 (> 10) ===")
    result2 = await branch_workflow(value=15).collect()
    print(f"Input: 15, Output: {result2.output}")
    
    # Test with workflow branching
    print("\n=== Testing workflow branching with value 7 ===")
    result3 = await branch_workflows(value=7).collect()
    print(f"Input: 7, Output: {result3.output}")

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Advanced conditional patterns

<CodeBlock language="python" code={`# Multiple condition branches
def multi_condition_branch(value: int) -> dict:
    """Handle multiple conditional branches."""
    if value < 0:
        return {"value": -1, "category": "negative"}
    elif value == 0:
        return {"value": 0, "category": "zero"}
    elif value <= 10:
        return {"value": 1, "category": "small"}
    elif value <= 100:
        return {"value": 2, "category": "medium"}
    else:
        return {"value": 3, "category": "large"}

# Conditional workflow with error handling
async def safe_conditional_workflow(value: int) -> dict:
    """Conditional workflow with error handling."""
    try:
        if value <= 10:
            result = await less_than_workflow(value=value).collect()
            return {"success": True, "data": result.output}
        else:
            result = await greater_than_workflow(value=value).collect()
            return {"success": True, "data": result.output}
    except Exception as e:
        return {"success": False, "error": str(e)}
`}/>

## Key differences from Mastra

1. **Branch Declaration**:
   - **Mastra**: Explicit `.branch([condition, step])` method with complex syntax
   - **Timbal**: Native Python `if/else` statements for natural conditional logic

2. **Condition Evaluation**:
   - **Mastra**: Complex async condition functions with `inputData` destructuring
   - **Timbal**: Simple Python boolean expressions and comparisons

3. **Workflow Structure**:
   - **Mastra**: `.branch([...]).commit()` with complex array syntax
   - **Timbal**: Natural Python control flow with workflow composition

4. **Data Handling**:
   - **Mastra**: Complex schema-based data passing between branches
   - **Timbal**: Direct function calls and return values

5. **Execution Model**:
   - **Mastra**: Workflow engine with explicit branching control
   - **Timbal**: Python-native conditional execution with workflow orchestration

The Timbal approach is much more intuitive and Pythonic - you use the same conditional logic you'd use in any Python program, making workflows easier to read, write, and maintain.

## Array as Input

Some workflows need to perform the same operation on every item in an array. This example demonstrates how to iterate over a list of strings and apply the same step to each one, producing a transformed array as the output.

## Repeating with array processing

In this example, the workflow processes each string in the input array by applying the `map_step` function to each item. For each item, it appends the text " mapStep" to the original value. After all items are processed, `step2` runs to pass the updated array to the output.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool
from typing import List

# Define the mapping step
def map_step(input_value: str) -> dict:
    """Adds mapStep suffix to input value."""
    return {
        "value": f"{input_value} mapStep"
    }

# Define step2 that processes the array
def step2(mapped_values: List[dict]) -> List[dict]:
    """Passes the mapped values to output."""
    return mapped_values

# Create the array processing workflow
def process_array(input_array: List[str]) -> List[dict]:
    """Process each item in the array using the map_step function."""
    # Apply map_step to each item in the array
    mapped_results = []
    for item in input_array:
        result = map_step(item)
        mapped_results.append(result)
    
    # Pass the results to step2
    return step2(mapped_results)

# Create the workflow
foreach_workflow = (
    Workflow(name="foreach-workflow")
    .step(process_array)
)

# Alternative: Using list comprehension for cleaner code
def process_array_clean(input_array: List[str]) -> List[dict]:
    """Process array using list comprehension."""
    # Apply map_step to each item using list comprehension
    mapped_results = [map_step(item) for item in input_array]
    return step2(mapped_results)

clean_foreach_workflow = (
    Workflow(name="clean-foreach-workflow")
    .step(process_array_clean)
)

# Alternative: Using Tool class for more complex array processing
def map_step_tool(input_value: str) -> dict:
    """Tool that adds mapStep suffix to input value."""
    return {
        "value": f"{input_value} mapStep"
    }

map_tool = Tool(
    name="map_step",
    description="adds mapStep suffix to input value",
    handler=map_step_tool
)

def process_array_with_tool(input_array: List[str]) -> List[dict]:
    """Process array using the Tool class."""
    mapped_results = []
    for item in input_array:
        result = map_tool(input_value=item)
        mapped_results.append(result)
    
    return step2(mapped_results)

tool_foreach_workflow = (
    Workflow(name="tool-foreach-workflow")
    .step(process_array_with_tool)
)
`}/>

## How array processing works in Timbal

1. **Array Input**: Workflows can accept lists as input parameters
2. **Iteration**: Use Python's native `for` loops or list comprehensions
3. **Step Application**: Apply the same function to each array item
4. **Result Collection**: Gather all results into a new array
5. **Output Processing**: Pass the processed array to subsequent steps

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    # Test with multiple string inputs
    input_array = ["hello", "world", "python", "workflow"]
    
    print("=== Testing foreach workflow ===")
    print(f"Input array: {input_array}")
    
    # Run the workflow
    result = await foreach_workflow(input_array=input_array).collect()
    
    print(f"Output: {result.output}")
    
    # Test the clean version
    print("\n=== Testing clean foreach workflow ===")
    result2 = await clean_foreach_workflow(input_array=input_array).collect()
    print(f"Output: {result2.output}")
    
    # Test with tool version
    print("\n=== Testing tool foreach workflow ===")
    result3 = await tool_foreach_workflow(input_array=input_array).collect()
    print(f"Output: {result3.output}")

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Advanced array processing patterns

<CodeBlock language="python" code={`# Parallel array processing
async def parallel_array_process(input_array: List[str]) -> List[dict]:
    """Process array items in parallel for better performance."""
    import asyncio
    
    # Create tasks for parallel processing
    tasks = [map_step(item) for item in input_array]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return step2(results)

# Array processing with filtering
def filter_and_map_array(input_array: List[str]) -> List[dict]:
    """Filter and map array items."""
    # Filter items that start with 'h' and then map them
    filtered_items = [item for item in input_array if item.startswith('h')]
    mapped_results = [map_step(item) for item in filtered_items]
    return step2(mapped_results)

# Array processing with error handling
def safe_array_process(input_array: List[str]) -> List[dict]:
    """Process array with error handling for individual items."""
    mapped_results = []
    errors = []
    
    for i, item in enumerate(input_array):
        try:
            result = map_step(item)
            mapped_results.append(result)
        except Exception as e:
            errors.append({"index": i, "item": item, "error": str(e)})
    
    if errors:
        print(f"Errors occurred: {errors}")
    
    return step2(mapped_results)
`}/>

## Key differences from Mastra

1. **Array Processing**:
   - **Mastra**: Explicit `.foreach(step)` method with complex workflow orchestration
   - **Timbal**: Native Python iteration with `for` loops and list comprehensions

2. **Data Flow**:
   - **Mastra**: Complex schema-based data transformation between steps
   - **Timbal**: Direct function calls and return values

3. **Workflow Structure**:
   - **Mastra**: `.foreach(step).then(step2).commit()` with complex chaining
   - **Timbal**: Natural Python array processing with workflow composition

4. **Performance**:
   - **Mastra**: Built-in parallel processing capabilities
   - **Timbal**: Can implement parallel processing using `asyncio.gather()`

5. **Flexibility**:
   - **Mastra**: Limited to the foreach framework
   - **Timbal**: Full Python language features including filtering, error handling, and custom logic

The Timbal approach gives you the flexibility to process arrays however you want - using standard Python patterns that you're already familiar with, while maintaining the workflow orchestration capabilities.
