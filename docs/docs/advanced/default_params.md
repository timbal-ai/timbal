---
title: Default Parameters
sidebar: 'docsSidebar'
---

# Default Parameters

Default parameters in Timbal allow you to set predefined values that are automatically injected into your runnable components, providing flexibility and reducing boilerplate code.

## Basic Usage

Default parameters are defined when creating a runnable and are merged with runtime parameters:

```python
from timbal.core_v2 import Tool

def analyze_data(data: str, method: str = "standard", threshold: float = 0.5):
    return f"Analyzed {data} using {method} with threshold {threshold}"

# Create tool with default parameters
tool = Tool(
    name="data_analyzer",
    handler=analyze_data,
    default_params={
        "method": "advanced",
        "threshold": 0.8
    }
)

# Execute with runtime parameters
result = await tool(data="sample_data").collect()
# Uses: method="advanced", threshold=0.8 (from defaults)
# Override: data="sample_data" (from runtime)

# Override default parameters
result = await tool(data="sample_data", method="simple").collect()
# Uses: method="simple" (overridden), threshold=0.8 (from defaults)
```

## Parameter Precedence

Runtime parameters always override default parameters:

```python
from timbal.core_v2 import Agent

# Agent with default parameters
agent = Agent(
    name="analysis_agent",
    model="gpt-4o-mini",
    default_params={
        "temperature": 0.3,
        "max_tokens": 500,
        "system_prompt": "You are a helpful analyst."
    }
)

# Runtime parameters override defaults
result = await agent(
    prompt="Analyze this data",
    temperature=0.7,  # Overrides default 0.3
    max_tokens=1000   # Overrides default 500
).collect()
```

## Dynamic Default Parameters

Default parameters can be computed dynamically using functions:

```python
from timbal.core_v2 import Tool
from datetime import datetime

def get_current_timestamp():
    return datetime.now().isoformat()

def process_data(data: str, timestamp: str, user_id: str = "default"):
    return f"Processed {data} at {timestamp} by {user_id}"

# Tool with dynamic default parameters
tool = Tool(
    name="data_processor",
    handler=process_data,
    default_params={
        "timestamp": get_current_timestamp,
        "user_id": "system_user"
    }
)

# The timestamp will be computed at execution time
result = await tool(data="important_data").collect()
```

## Context-Aware Default Parameters

Default parameters can be adapted based on execution context:

```python
from timbal.core_v2 import Agent
from timbal.state import get_run_context

async def context_aware_defaults(input_dict):
    """Hook that sets default parameters based on context."""
    run_context = get_run_context()
    
    # Set defaults based on user preferences
    user_preferences = run_context.data.get('user_preferences', {})
    
    if user_preferences.get('detailed_responses'):
        input_dict.setdefault('max_tokens', 2000)
        input_dict.setdefault('temperature', 0.1)
    else:
        input_dict.setdefault('max_tokens', 500)
        input_dict.setdefault('temperature', 0.7)
    
    # Set language based on context
    language = run_context.data.get('language', 'english')
    if language == 'spanish':
        input_dict.setdefault('system_prompt', "Responde siempre en español.")
    else:
        input_dict.setdefault('system_prompt', "Always respond in English.")

# Agent with context-aware defaults
agent = Agent(
    name="adaptive_agent",
    model="gpt-4o-mini",
    pre_hook=context_aware_defaults
)

# Execute with different contexts
context = RunContext()
context.data['user_preferences'] = {'detailed_responses': True}
context.data['language'] = 'spanish'

result = await agent(
    prompt="Explain quantum computing",
    context=context
).collect()
```

## Environment-Based Defaults

Default parameters can be set based on environment variables:

```python
from timbal.core_v2 import Tool
import os

def api_call(endpoint: str, api_key: str, timeout: int = 30):
    return f"Calling {endpoint} with timeout {timeout}"

# Tool with environment-based defaults
tool = Tool(
    name="api_client",
    handler=api_call,
    default_params={
        "api_key": os.getenv("API_KEY", "default_key"),
        "timeout": int(os.getenv("API_TIMEOUT", "60"))
    }
)

# Uses environment variables or fallback values
result = await tool(endpoint="/data").collect()
```

## Conditional Default Parameters

Default parameters can be set conditionally based on input:

```python
from timbal.core_v2 import Tool

def process_image(image_data: str, format: str, quality: int = 90):
    return f"Processed image in {format} format with quality {quality}"

async def conditional_defaults(input_dict):
    """Hook that sets defaults based on input data."""
    image_data = input_dict.get('image_data', '')
    
    # Set format based on image data
    if image_data.startswith('data:image/jpeg'):
        input_dict.setdefault('format', 'jpeg')
        input_dict.setdefault('quality', 85)
    elif image_data.startswith('data:image/png'):
        input_dict.setdefault('format', 'png')
        input_dict.setdefault('quality', 95)
    else:
        input_dict.setdefault('format', 'auto')
        input_dict.setdefault('quality', 90)

# Tool with conditional defaults
tool = Tool(
    name="image_processor",
    handler=process_image,
    pre_hook=conditional_defaults
)

# Defaults will be set based on image data
result = await tool(image_data="data:image/jpeg;base64,...").collect()
```

## Nested Default Parameters

Default parameters work with nested runnables:

```python
from timbal.core_v2 import Agent, Tool

def search_tool(query: str, limit: int = 10, source: str = "web"):
    return f"Searching for '{query}' with limit {limit} from {source}"

def analyze_tool(data: str, method: str = "standard", confidence: float = 0.8):
    return f"Analyzing {data} using {method} with confidence {confidence}"

# Create tools with defaults
search_tool = Tool(
    name="search",
    handler=search_tool,
    default_params={"limit": 20, "source": "database"}
)

analyze_tool = Tool(
    name="analyze",
    handler=analyze_tool,
    default_params={"method": "advanced", "confidence": 0.9}
)

# Agent with tools that have their own defaults
agent = Agent(
    name="research_agent",
    model="gpt-4o-mini",
    tools=[search_tool, analyze_tool],
    default_params={
        "temperature": 0.3,
        "max_tokens": 1000
    }
)

# All defaults are respected unless overridden
result = await agent(
    prompt="Research machine learning trends"
).collect()
```

## Validation with Default Parameters

Default parameters are validated against the handler's parameter model:

```python
from timbal.core_v2 import Tool
from timbal.types import Field

def validated_function(
    text: str = Field(description="Text to process"),
    max_length: int = Field(default=100, ge=1, le=1000),
    case_sensitive: bool = Field(default=True)
):
    return f"Processed: {text[:max_length]} (case_sensitive={case_sensitive})"

# Tool with validated defaults
tool = Tool(
    name="text_processor",
    handler=validated_function,
    default_params={
        "max_length": 200,  # Must be between 1 and 1000
        "case_sensitive": False
    }
)

# Invalid defaults would raise validation errors
# tool = Tool(
#     name="invalid_tool",
#     handler=validated_function,
#     default_params={"max_length": 2000}  # Would fail validation
# )
```

## Complete Example: Smart Default System

```python
from timbal.core_v2 import Agent, Tool
from timbal.state import RunContext, get_run_context
import os
from datetime import datetime

class SmartDefaults:
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.user_tier = os.getenv("USER_TIER", "basic")
    
    async def smart_defaults_hook(self, input_dict):
        """Comprehensive default parameter system."""
        run_context = get_run_context()
        
        # Environment-based defaults
        if self.environment == "production":
            input_dict.setdefault("max_tokens", 1000)
            input_dict.setdefault("temperature", 0.1)
        else:
            input_dict.setdefault("max_tokens", 2000)
            input_dict.setdefault("temperature", 0.7)
        
        # User tier-based defaults
        if self.user_tier == "premium":
            input_dict.setdefault("model", "gpt-4o")
        else:
            input_dict.setdefault("model", "gpt-4o-mini")
        
        # Context-based defaults
        if run_context:
            user_preferences = run_context.data.get('user_preferences', {})
            if user_preferences.get('detailed_mode'):
                input_dict.setdefault("max_tokens", input_dict.get("max_tokens", 1000) * 2)
        
        # Time-based defaults
        hour = datetime.now().hour
        if 22 <= hour or hour <= 6:
            input_dict.setdefault("system_prompt", "It's late. Keep responses concise.")

# Create smart agent
smart_defaults = SmartDefaults()
agent = Agent(
    name="smart_agent",
    model="gpt-4o-mini",
    pre_hook=smart_defaults.smart_defaults_hook,
    default_params={
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.5
    }
)

# Execute with smart defaults
context = RunContext()
context.data['user_preferences'] = {'detailed_mode': True}

result = await agent(
    prompt="Explain artificial intelligence",
    context=context
).collect()
```

Default parameters provide a powerful way to configure your runnables with sensible defaults while maintaining the flexibility to override them when needed.
