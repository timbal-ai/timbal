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

<CodeBlock language="python" code={`from timbal.core_v2 import Tool
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

<CodeBlock language="python" code={`from timbal.core_v2 import Tool

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

<CodeBlock language="python" code={`from timbal.core_v2 import Agent
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

<CodeBlock language="python" code={`from timbal.core_v2 import Tool
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

<CodeBlock language="python" code={`from timbal.core_v2 import Tool
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

## Performance Monitoring

Hooks can be used for performance monitoring and logging:

<CodeBlock language="python" code={`from timbal.core_v2 import Tool
from timbal.state import get_run_context
import time

async def performance_pre_hook(input_dict):
    """Pre-hook for performance monitoring."""
    run_context = get_run_context()
    run_context.data['start_time'] = time.time()
    run_context.data['input_size'] = len(str(input_dict))

async def performance_post_hook(output):
    """Post-hook for performance monitoring."""
    run_context = get_run_context()
    
    end_time = time.time()
    start_time = run_context.data.get('start_time', end_time)
    execution_time = end_time - start_time
    
    # Log performance metrics
    print(f"Execution time: {execution_time:.3f}s")
    print(f"Input size: {run_context.data.get('input_size', 0)} chars")
    print(f"Output type: {type(output).__name__}")
    
    # Store metrics in context
    run_context.data['execution_time'] = execution_time
    run_context.data['output_type'] = type(output).__name__

def expensive_operation(data: str):
    import time
    time.sleep(0.1)  # Simulate expensive operation
    return f"Processed: {data}"

# Tool with performance monitoring
tool = Tool(
    name="expensive_processor",
    handler=expensive_operation,
    pre_hook=performance_pre_hook,
    post_hook=performance_post_hook
)

# Performance will be monitored
result = await tool(data="large dataset").collect()`}/>

## Conditional Processing

Hooks can implement conditional logic based on input or context:

<CodeBlock language="python" code={`from timbal.core_v2 import Agent
from timbal.state import get_run_context

async def conditional_pre_hook(input_dict):
    """Pre-hook with conditional processing."""
    run_context = get_run_context()
    
    # Check user permissions
    user_role = run_context.data.get('user_role', 'user')
    if user_role == 'admin':
        input_dict['admin_mode'] = True
        input_dict['max_tokens'] = 4000
    else:
        input_dict['admin_mode'] = False
        input_dict['max_tokens'] = 1000
    
    # Check content type
    prompt = input_dict.get('prompt', '')
    if any(word in prompt.lower() for word in ['code', 'programming', 'technical']):
        input_dict['system_prompt'] = "You are a technical expert. Provide detailed code examples."
    elif any(word in prompt.lower() for word in ['business', 'strategy', 'market']):
        input_dict['system_prompt'] = "You are a business consultant. Provide strategic insights."
    else:
        input_dict['system_prompt'] = "You are a helpful assistant."

async def conditional_post_hook(output):
    """Post-hook with conditional processing."""
    run_context = get_run_context()
    
    # Add different metadata based on user role
    user_role = run_context.data.get('user_role', 'user')
    if user_role == 'admin':
        output.admin_metadata = {
            'processed_by_admin': True,
            'full_access': True
        }
    else:
        output.user_metadata = {
            'processed_by_user': True,
            'limited_access': True
        }

# Agent with conditional processing
agent = Agent(
    name="conditional_agent",
    model="gpt-4o-mini",
    pre_hook=conditional_pre_hook,
    post_hook=conditional_post_hook
)

# Execute with different roles
context = RunContext()
context.data['user_role'] = 'admin'

result = await agent(
    prompt="Write a Python function for data analysis",
    context=context
).collect()`}/>

## Error Handling in Hooks

Hooks can implement custom error handling:

<CodeBlock language="python" code={`from timbal.core_v2 import Tool

async def error_handling_pre_hook(input_dict):
    """Pre-hook with error handling."""
    try:
        # Validate critical parameters
        if 'api_key' in input_dict:
            if not input_dict['api_key'] or input_dict['api_key'] == 'invalid':
                raise ValueError("Invalid API key")
        
        # Set fallback values
        if 'timeout' not in input_dict:
            input_dict['timeout'] = 30
        
    except Exception as e:
        # Log error and set default values
        print(f"Pre-hook error: {e}")
        input_dict['error_handled'] = True
        input_dict['fallback_mode'] = True

async def error_handling_post_hook(output):
    """Post-hook with error handling."""
    try:
        # Check for errors in output
        if hasattr(output, 'error') and output.error:
            print(f"Output error detected: {output.error}")
            # Transform error into user-friendly message
            output.user_message = "An error occurred, but we've handled it gracefully."
        
    except Exception as e:
        print(f"Post-hook error: {e}")

def api_call(endpoint: str, api_key: str, timeout: int = 30):
    if api_key == 'invalid':
        raise ValueError("API call failed")
    return f"Successfully called {endpoint}"

# Tool with error handling
tool = Tool(
    name="api_client",
    handler=api_call,
    pre_hook=error_handling_pre_hook,
    post_hook=error_handling_post_hook
)

# Error will be handled gracefully
result = await tool(
    endpoint="/data",
    api_key="invalid"
).collect()`}/>

## Complete Example: Advanced Hook System

<CodeBlock language="python" code={`from timbal.core_v2 import Agent, Tool
from timbal.state import RunContext, get_run_context
import time
import json
from datetime import datetime

class AdvancedHookSystem:
    def __init__(self):
        self.request_count = 0
    
    async def advanced_pre_hook(self, input_dict):
        """Comprehensive pre-hook system."""
        run_context = get_run_context()
        self.request_count += 1
        
        # Performance tracking
        run_context.data['start_time'] = time.time()
        run_context.data['request_id'] = f"req_{self.request_count}"
        
        # Input validation and sanitization
        if 'prompt' in input_dict:
            prompt = input_dict['prompt']
            # Sanitize prompt
            prompt = prompt.strip()[:2000]  # Limit length
            input_dict['prompt'] = prompt
        
        # Context-aware parameter setting
        user_preferences = run_context.data.get('user_preferences', {})
        if user_preferences.get('detailed_mode'):
            input_dict.setdefault('max_tokens', 2000)
            input_dict.setdefault('temperature', 0.1)
        else:
            input_dict.setdefault('max_tokens', 500)
            input_dict.setdefault('temperature', 0.7)
        
        # Add metadata
        input_dict['processed_at'] = datetime.now().isoformat()
        input_dict['hook_version'] = "2.0"
    
    async def advanced_post_hook(self, output):
        """Comprehensive post-hook system."""
        run_context = get_run_context()
        
        # Performance calculation
        start_time = run_context.data.get('start_time', time.time())
        execution_time = time.time() - start_time
        
        # Add performance metadata
        if hasattr(output, 'metadata'):
            output.metadata = {
                'execution_time': execution_time,
                'request_id': run_context.data.get('request_id'),
                'processed_at': datetime.now().isoformat(),
                'hook_version': "2.0"
            }
        
        # Log performance
        print(f"Request {run_context.data.get('request_id')} completed in {execution_time:.3f}s")
        
        # Store metrics
        run_context.data['last_execution_time'] = execution_time
        run_context.data['total_requests'] = self.request_count

# Create advanced system
hook_system = AdvancedHookSystem()

# Agent with advanced hooks
agent = Agent(
    name="advanced_agent",
    model="gpt-4o-mini",
    pre_hook=hook_system.advanced_pre_hook,
    post_hook=hook_system.advanced_post_hook
)

# Execute with advanced hooks
context = RunContext()
context.data['user_preferences'] = {'detailed_mode': True}

result = await agent(
    prompt="Explain quantum computing in detail",
    context=context
).collect()`}/>

Hooks provide a powerful way to add middleware functionality to your Timbal components, enabling input/output transformation, validation, monitoring, and context-aware behavior.
