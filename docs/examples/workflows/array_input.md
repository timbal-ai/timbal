---
title: Array as Input
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

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

# Array processing with conditional logic
def conditional_array_process(input_array: List[str], threshold: int = 3) -> List[dict]:
    """Process array with conditional logic based on item length."""
    mapped_results = []
    
    for item in input_array:
        if len(item) > threshold:
            # Long items get the full map_step treatment
            result = map_step(item)
        else:
            # Short items get a simplified treatment
            result = {"value": f"{item} (short)"}
        
        mapped_results.append(result)
    
    return step2(mapped_results)

# Array processing with batching
def batched_array_process(input_array: List[str], batch_size: int = 2) -> List[dict]:
    """Process array in batches for memory efficiency."""
    mapped_results = []
    
    for i in range(0, len(input_array), batch_size):
        batch = input_array[i:i + batch_size]
        batch_results = [map_step(item) for item in batch]
        mapped_results.extend(batch_results)
        
        print(f"Processed batch {i//batch_size + 1}: {batch}")
    
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

The Timbal approach provides:
- **Multiple Implementation Options**: Functions, list comprehensions, and Tool class approaches
- **Advanced Features**: Parallel processing, filtering, error handling, conditional logic, and batching
- **Native Python Patterns**: Use the same array processing techniques you know from Python
- **Clean Code**: List comprehensions for concise array transformations
- **Workflow Integration**: Seamlessly integrate array processing into your workflows

This makes array processing much more intuitive and powerful while maintaining all the workflow orchestration capabilities you need!
