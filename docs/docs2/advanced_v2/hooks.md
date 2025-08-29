---
title: Pre and Post Hooks
sidebar: 'docsSidebar_v2'
---

import CodeBlock from '@site/src/theme/CodeBlock';

# Pre and Post Hooks

Pre and post hooks in Timbal provide powerful middleware-style functionality that allows you to intercept and modify inputs and outputs during execution. These hooks follow an in-place modification pattern and are essential for building adaptive and context-aware systems.

## Hook Basics

Hooks are functions that receive mutable references and modify them in-place. No return value is expected - all changes happen by mutating the passed objects.

### Pre-hook

Pre-hooks execute before the main handler and can modify input parameters:

<CodeBlock language="python" code={`from timbal.core import Tool
from timbal.state import get_run_context

async def pre_hook_example(input_dict):
    """Pre-hook that modifies input parameters."""
    # Read data
    original_text = input_dict.get('text', '')
    
    # Modify data
    input_dict['text'] = f"Enhanced: {original_text}"
    
    # Add new data
    input_dict['timestamp'] = "2024-01-01T00:00:00Z"
    
    # Remove unwanted data
    if 'debug_info' in input_dict:
        del input_dict['debug_info']

def process_text(text: str, timestamp: str = None):
    return f"Processed: {text} at {timestamp}"

# Tool with pre-hook
tool = Tool(
    name="text_processor",
    handler=process_text,
    pre_hook=pre_hook_example
)

# Execute - pre-hook will modify the input
result = await tool(text="Hello world").collect()
# Output: "Processed: Enhanced: Hello world at 2024-01-01T00:00:00Z"`}/>

### Post-hook

Post-hooks execute after the main handler and can modify the output:

<CodeBlock language="python" code={`from timbal.core import Tool

async def post_hook_example(output):
    """Post-hook that modifies the output."""
    if isinstance(output, str):
        # Modify string output
        output = f"Final result: {output.upper()}"
    elif isinstance(output, list):
        # Modify list output
        output.append("Additional item")
    elif isinstance(output, dict):
        # Modify dict output
        output['processed_at'] = "2024-01-01T00:00:00Z"
        output['status'] = 'completed'

def generate_data():
    return {"data": "sample", "count": 42}

# Tool with post-hook
tool = Tool(
    name="data_generator",
    handler=generate_data,
    post_hook=post_hook_example
)

# Execute - post-hook will modify the output
result = await tool().collect()
# Output: {"data": "sample", "count": 42, "processed_at": "2024-01-01T00:00:00Z", "status": "completed"}`}/>

## Context Access in Hooks

Hooks can access the run context to share data and make decisions:

<CodeBlock language="python" code={`from timbal.core import Agent
from timbal.state import get_run_context, RunContext

async def context_aware_pre_hook(input_dict):
    """Pre-hook that uses run context."""
    run_context = get_run_context()
    
    # Store data for other hooks
    run_context.data['pre_hook_executed'] = True
    run_context.data['input_keys'] = list(input_dict.keys())
    
    # Read data from context
    user_id = run_context.data.get('user_id', 'anonymous')
    user_preferences = run_context.data.get('user_preferences', {})
    
    # Modify input based on context
    if user_preferences.get('detailed_mode'):
        input_dict['max_tokens'] = 2000
        input_dict['temperature'] = 0.1
    else:
        input_dict['max_tokens'] = 500
        input_dict['temperature'] = 0.7
    
    # Add user context
    input_dict['user_id'] = user_id

async def context_aware_post_hook(output):
    """Post-hook that reads from run context."""
    run_context = get_run_context()
    
    # Read data stored by pre-hook
    if run_context.data.get('pre_hook_executed'):
        run_context.data['post_hook_found_pre_data'] = True
    
    # Add metadata to output
    if hasattr(output, 'content'):
        output.metadata = {
            'processed_by': run_context.data.get('user_id', 'system'),
            'execution_time': run_context.data.get('execution_time', 0)
        }

# Agent with context-aware hooks
agent = Agent(
    name="context_agent",
    model="gpt-4o-mini",
    pre_hook=context_aware_pre_hook,
    post_hook=context_aware_post_hook
)

# Execute with context
context = RunContext()
context.data['user_id'] = 'user123'
context.data['user_preferences'] = {'detailed_mode': True}

result = await agent(
    prompt="Explain machine learning",
    context=context
).collect()`}/>

## Input Validation and Sanitization

Pre-hooks are perfect for input validation and sanitization:

<CodeBlock language="python" code={`from timbal.core import Tool
import re

async def validate_and_sanitize(input_dict):
    """Pre-hook for input validation and sanitization."""
    # Validate required fields
    if 'email' in input_dict:
        email = input_dict['email']
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            raise ValueError("Invalid email format")
    
    # Sanitize text inputs
    if 'text' in input_dict:
        text = input_dict['text']
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Limit length
        text = text[:1000]
        input_dict['text'] = text
    
    # Normalize numeric inputs
    if 'age' in input_dict:
        age = input_dict['age']
        try:
            age = int(age)
            if age < 0 or age > 150:
                raise ValueError("Age out of valid range")
            input_dict['age'] = age
        except (ValueError, TypeError):
            raise ValueError("Invalid age value")

def process_user_data(name: str, email: str, text: str, age: int):
    return f"Processed data for {name} ({email}), age {age}: {text}"

# Tool with validation
tool = Tool(
    name="user_processor",
    handler=process_user_data,
    pre_hook=validate_and_sanitize
)

# This will be validated and sanitized
result = await tool(
    name="John Doe",
    email="john@example.com",
    text="<script>alert('xss')</script>Hello world!",
    age="25"
).collect()`}/>

## Output Transformation

Post-hooks are ideal for output transformation and formatting:

<CodeBlock language="python" code={`from timbal.core import Tool
import json

async def format_output(output):
    """Post-hook for output formatting."""
    if isinstance(output, dict):
        # Add metadata
        output['formatted_at'] = "2024-01-01T00:00:00Z"
        output['version'] = "1.0"
        
        # Format nested data
        if 'data' in output:
            if isinstance(output['data'], list):
                output['data_count'] = len(output['data'])
                output['data'] = output['data'][:10]  # Limit to 10 items
        
        # Convert to JSON string for API responses
        output = json.dumps(output, indent=2)
    
    elif isinstance(output, str):
        # Add formatting to text output
        output = f"=== RESULT ===\n{output}\n=== END ==="
    
    return output

def fetch_data(query: str):
    return {
        "query": query,
        "data": [f"item_{i}" for i in range(15)],
        "status": "success"
    }

# Tool with output formatting
tool = Tool(
    name="data_fetcher",
    handler=fetch_data,
    post_hook=format_output
)

# Output will be formatted
result = await tool(query="test").collect()`}/>


Hooks provide a powerful way to add middleware functionality to your Timbal components, enabling input/output transformation, validation, monitoring, and context-aware behavior.
