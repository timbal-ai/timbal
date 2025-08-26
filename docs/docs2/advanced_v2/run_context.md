---
title: Run Context and Data Sharing
sidebar: 'docsSidebar_v2'
---

import CodeBlock from '@site/src/theme/CodeBlock';

# Run Context and Data Sharing

The Run Context in Timbal provides a powerful mechanism for sharing data across the execution of agents and workflows. It enables state management, data persistence, and communication between different components in your AI system.

## Understanding Run Context

The `RunContext` is a shared data container that persists throughout the execution of a runnable and is accessible to all nested components:

<CodeBlock language="python" code={`
from timbal.state import RunContext, get_run_context

# Create a run context
context = RunContext()
print(f"Context ID: {context.id}")
print(f"Parent ID: {context.parent_id}")

# Access the current context
current_context = get_run_context()
if current_context:
    print(f"Current context ID: {current_context.id}")
`}/>

## Basic Data Sharing

The run context provides a simple key-value store for sharing data:

<CodeBlock language="python" code={`
from timbal.core import Agent, Tool
from timbal.state import RunContext, get_run_context

def data_collector(query: str):
    """Tool that collects data and stores it in context."""
    run_context = get_run_context()
    
    # Store data in context
    run_context.data['collected_data'] = f"Data for: {query}"
    run_context.data['collection_timestamp'] = "2024-01-01T00:00:00Z"
    
    return f"Collected data for: {query}"

def data_processor():
    """Tool that processes data from context."""
    run_context = get_run_context()
    
    # Retrieve data from context
    collected_data = run_context.data.get('collected_data', 'No data')
    timestamp = run_context.data.get('collection_timestamp', 'Unknown')
    
    return f"Processed: {collected_data} (collected at {timestamp})"

# Create tools
collector_tool = Tool(name="collector", handler=data_collector)
processor_tool = Tool(name="processor", handler=data_processor)

# Create agent with tools
agent = Agent(
    name="data_agent",
    model="gpt-4o-mini",
    tools=[collector_tool, processor_tool]
)

# Execute with context
context = RunContext()
result = await agent(
    prompt="Collect and process data about machine learning",
    context=context
).collect()

# Data persists in context
print(f"Context data: {context.data}")
`}/>

## Context Inheritance

Run contexts can be nested, with child contexts inheriting from parent contexts:

<CodeBlock language="python" code={`
from timbal.state import RunContext, get_run_context

# Create parent context
parent_context = RunContext()
parent_context.data['parent_data'] = "Shared from parent"
parent_context.data['user_id'] = "user123"

# Create child context
child_context = RunContext(parent_id=parent_context.id)
child_context.data['child_data'] = "Specific to child"

# In a nested execution, the child context would have access to parent data
print(f"Parent data: {parent_context.data}")
print(f"Child data: {child_context.data}")
`}/>

## Cross-Component Communication

The run context enables communication between different components:

<CodeBlock language="python" code={`
from timbal.core import Agent, Tool
from timbal.state import get_run_context

def authentication_tool(user_id: str):
    """Tool that authenticates user and stores info in context."""
    run_context = get_run_context()
    
    # Simulate authentication
    if user_id == "admin":
        run_context.data['user_role'] = "admin"
        run_context.data['permissions'] = ["read", "write", "delete"]
    else:
        run_context.data['user_role'] = "user"
        run_context.data['permissions'] = ["read"]
    
    run_context.data['authenticated_user'] = user_id
    return f"Authenticated user: {user_id}"

def permission_check_tool(action: str):
    """Tool that checks permissions from context."""
    run_context = get_run_context()
    
    user_role = run_context.data.get('user_role', 'anonymous')
    permissions = run_context.data.get('permissions', [])
    
    if action in permissions:
        return f"Permission granted for {action} (role: {user_role})"
    else:
        return f"Permission denied for {action} (role: {user_role})"

def data_access_tool(resource: str):
    """Tool that accesses data based on permissions."""
    run_context = get_run_context()
    
    user_role = run_context.data.get('user_role', 'anonymous')
    permissions = run_context.data.get('permissions', [])
    
    if 'read' in permissions:
        return f"Accessing {resource} as {user_role}"
    else:
        return f"Access denied to {resource}"

# Create tools
auth_tool = Tool(name="authenticate", handler=authentication_tool)
permission_tool = Tool(name="check_permission", handler=permission_check_tool)
access_tool = Tool(name="access_data", handler=data_access_tool)

# Create agent
agent = Agent(
    name="secure_agent",
    model="gpt-4o-mini",
    tools=[auth_tool, permission_tool, access_tool]
)

# Execute with authentication
context = RunContext()
result = await agent(
    prompt="Authenticate as admin and access sensitive data",
    context=context
).collect()

# Context contains authentication state
print(f"User role: {context.data.get('user_role')}")
print(f"Permissions: {context.data.get('permissions')}")
`}/>

## State Management

The run context is ideal for managing application state:

<CodeBlock language="python" code={`
from timbal.core import Agent, Tool
from timbal.state import get_run_context
import time

class SessionManager:
    def __init__(self):
        self.session_data = {}
    
    def start_session(self, user_id: str):
        """Tool that starts a new session."""
        run_context = get_run_context()
        
        session_id = f"session_{int(time.time())}"
        run_context.data['session_id'] = session_id
        run_context.data['user_id'] = user_id
        run_context.data['session_start'] = time.time()
        run_context.data['interaction_count'] = 0
        
        return f"Started session {session_id} for user {user_id}"
    
    def track_interaction(self):
        """Tool that tracks interactions."""
        run_context = get_run_context()
        
        count = run_context.data.get('interaction_count', 0)
        run_context.data['interaction_count'] = count + 1
        run_context.data['last_interaction'] = time.time()
        
        return f"Interaction {count + 1} tracked"
    
    def get_session_info(self):
        """Tool that retrieves session information."""
        run_context = get_run_context()
        
        session_id = run_context.data.get('session_id', 'No session')
        user_id = run_context.data.get('user_id', 'Unknown')
        interaction_count = run_context.data.get('interaction_count', 0)
        session_start = run_context.data.get('session_start', 0)
        
        session_duration = time.time() - session_start if session_start > 0 else 0
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "interaction_count": interaction_count,
            "session_duration": f"{session_duration:.2f}s"
        }

# Create session manager
session_manager = SessionManager()

# Create tools
start_session_tool = Tool(name="start_session", handler=session_manager.start_session)
track_interaction_tool = Tool(name="track_interaction", handler=session_manager.track_interaction)
get_session_info_tool = Tool(name="get_session_info", handler=session_manager.get_session_info)

# Create agent
agent = Agent(
    name="session_agent",
    model="gpt-4o-mini",
    tools=[start_session_tool, track_interaction_tool, get_session_info_tool]
)

# Execute with session management
context = RunContext()
result = await agent(
    prompt="Start a session for user123 and track this interaction",
    context=context
).collect()

# Session state is maintained in context
print(f"Session ID: {context.data.get('session_id')}")
print(f"Interaction count: {context.data.get('interaction_count')}")
`}/>

## Configuration Management

The run context can store configuration that affects behavior:

<CodeBlock language="python" code={`
from timbal.core import Agent, Tool
from timbal.state import get_run_context

def set_config_tool(config_type: str, value: str):
    """Tool that sets configuration in context."""
    run_context = get_run_context()
    
    if config_type == "language":
        run_context.data['config.language'] = value
        return f"Language set to: {value}"
    elif config_type == "detail_level":
        run_context.data['config.detail_level'] = value
        return f"Detail level set to: {value}"
    elif config_type == "timezone":
        run_context.data['config.timezone'] = value
        return f"Timezone set to: {value}"
    else:
        return f"Unknown config type: {config_type}"

def get_config_tool(config_type: str):
    """Tool that retrieves configuration from context."""
    run_context = get_run_context()
    
    config_key = f"config.{config_type}"
    value = run_context.data.get(config_key, "Not set")
    
    return f"{config_type}: {value}"

def adaptive_response_tool(prompt: str):
    """Tool that adapts response based on configuration."""
    run_context = get_run_context()
    
    language = run_context.data.get('config.language', 'english')
    detail_level = run_context.data.get('config.detail_level', 'medium')
    
    if detail_level == 'high':
        response = f"Detailed response in {language}: {prompt}"
    elif detail_level == 'low':
        response = f"Brief response in {language}: {prompt}"
    else:
        response = f"Standard response in {language}: {prompt}"
    
    return response

# Create tools
set_config_tool = Tool(name="set_config", handler=set_config_tool)
get_config_tool = Tool(name="get_config", handler=get_config_tool)
adaptive_response_tool = Tool(name="adaptive_response", handler=adaptive_response_tool)

# Create agent
agent = Agent(
    name="configurable_agent",
    model="gpt-4o-mini",
    tools=[set_config_tool, get_config_tool, adaptive_response_tool]
)

# Execute with configuration
context = RunContext()
result = await agent(
    prompt="Set language to Spanish, detail level to high, and respond to 'Hello'",
    context=context
).collect()

# Configuration is stored in context
print(f"Language: {context.data.get('config.language')}")
print(f"Detail level: {context.data.get('config.detail_level')}")
`}/>

## Performance Tracking

The run context can track performance metrics across components:

<CodeBlock language="python" code={`
from timbal.core import Agent, Tool
from timbal.state import get_run_context
import time

def start_timer_tool(operation_name: str):
    """Tool that starts timing an operation."""
    run_context = get_run_context()
    
    start_time = time.time()
    run_context.data[f'timer.{operation_name}.start'] = start_time
    
    return f"Started timer for: {operation_name}"

def end_timer_tool(operation_name: str):
    """Tool that ends timing an operation."""
    run_context = get_run_context()
    
    start_time = run_context.data.get(f'timer.{operation_name}.start', 0)
    end_time = time.time()
    duration = end_time - start_time
    
    run_context.data[f'timer.{operation_name}.end'] = end_time
    run_context.data[f'timer.{operation_name}.duration'] = duration
    
    # Track total execution time
    total_time = run_context.data.get('total_execution_time', 0)
    run_context.data['total_execution_time'] = total_time + duration
    
    return f"Operation {operation_name} took {duration:.3f}s"

def get_performance_report_tool():
    """Tool that generates a performance report."""
    run_context = get_run_context()
    
    total_time = run_context.data.get('total_execution_time', 0)
    timers = {}
    
    # Collect all timer data
    for key, value in run_context.data.items():
        if key.startswith('timer.') and key.endswith('.duration'):
            operation = key.split('.')[1]
            timers[operation] = value
    
    return {
        "total_execution_time": f"{total_time:.3f}s",
        "operation_timers": timers,
        "operation_count": len(timers)
    }

# Create tools
start_timer_tool = Tool(name="start_timer", handler=start_timer_tool)
end_timer_tool = Tool(name="end_timer", handler=end_timer_tool)
get_performance_report_tool = Tool(name="get_performance_report", handler=get_performance_report_tool)

# Create agent
agent = Agent(
    name="performance_agent",
    model="gpt-4o-mini",
    tools=[start_timer_tool, end_timer_tool, get_performance_report_tool]
)

# Execute with performance tracking
context = RunContext()
result = await agent(
    prompt="Start timer for 'data_processing', wait 2 seconds, end timer, and get performance report",
    context=context
).collect()

# Performance data is available in context
print(f"Total execution time: {context.data.get('total_execution_time', 0):.3f}s")
`}/>

## Complete Example: Advanced Context System

<CodeBlock language="python" code={`
from timbal.core import Agent, Tool
from timbal.state import RunContext, get_run_context
import time
import json
from datetime import datetime

class AdvancedContextSystem:
    def __init__(self):
        self.request_id = 0
    
    def initialize_context_tool(self, user_id: str, session_type: str):
        """Tool that initializes a comprehensive context."""
        run_context = get_run_context()
        self.request_id += 1
        
        # Basic session info
        run_context.data['session.user_id'] = user_id
        run_context.data['session.type'] = session_type
        run_context.data['session.request_id'] = f"req_{self.request_id}"
        run_context.data['session.start_time'] = time.time()
        
        # Performance tracking
        run_context.data['performance.start_time'] = time.time()
        run_context.data['performance.operations'] = []
        
        # Configuration defaults
        run_context.data['config.language'] = 'english'
        run_context.data['config.detail_level'] = 'medium'
        run_context.data['config.timezone'] = 'UTC'
        
        # State tracking
        run_context.data['state.interaction_count'] = 0
        run_context.data['state.last_activity'] = time.time()
        
        return f"Initialized context for user {user_id} (session: {session_type})"
    
    def update_activity_tool(self):
        """Tool that updates activity tracking."""
        run_context = get_run_context()
        
        count = run_context.data.get('state.interaction_count', 0)
        run_context.data['state.interaction_count'] = count + 1
        run_context.data['state.last_activity'] = time.time()
        
        return f"Activity updated (interaction {count + 1})"
    
    def set_config_tool(self, config_type: str, value: str):
        """Tool that sets configuration."""
        run_context = get_run_context()
        
        config_key = f"config.{config_type}"
        old_value = run_context.data.get(config_key, "Not set")
        run_context.data[config_key] = value
        
        return f"Config updated: {config_type} = {value} (was: {old_value})"
    
    def track_operation_tool(self, operation_name: str, duration: float):
        """Tool that tracks operation performance."""
        run_context = get_run_context()
        
        operations = run_context.data.get('performance.operations', [])
        operations.append({
            'name': operation_name,
            'duration': duration,
            'timestamp': time.time()
        })
        run_context.data['performance.operations'] = operations
        
        return f"Tracked operation: {operation_name} ({duration:.3f}s)"
    
    def get_context_report_tool(self):
        """Tool that generates a comprehensive context report."""
        run_context = get_run_context()
        
        session_start = run_context.data.get('session.start_time', 0)
        session_duration = time.time() - session_start if session_start > 0 else 0
        
        return {
            "session": {
                "user_id": run_context.data.get('session.user_id'),
                "type": run_context.data.get('session.type'),
                "request_id": run_context.data.get('session.request_id'),
                "duration": f"{session_duration:.2f}s"
            },
            "state": {
                "interaction_count": run_context.data.get('state.interaction_count', 0),
                "last_activity": run_context.data.get('state.last_activity', 0)
            },
            "config": {
                "language": run_context.data.get('config.language'),
                "detail_level": run_context.data.get('config.detail_level'),
                "timezone": run_context.data.get('config.timezone')
            },
            "performance": {
                "total_operations": len(run_context.data.get('performance.operations', [])),
                "operations": run_context.data.get('performance.operations', [])
            }
        }

# Create advanced system
context_system = AdvancedContextSystem()

# Create tools
init_context_tool = Tool(name="initialize_context", handler=context_system.initialize_context_tool)
update_activity_tool = Tool(name="update_activity", handler=context_system.update_activity_tool)
set_config_tool = Tool(name="set_config", handler=context_system.set_config_tool)
track_operation_tool = Tool(name="track_operation", handler=context_system.track_operation_tool)
get_context_report_tool = Tool(name="get_context_report", handler=context_system.get_context_report_tool)

# Create agent
agent = Agent(
    name="advanced_context_agent",
    model="gpt-4o-mini",
    tools=[init_context_tool, update_activity_tool, set_config_tool, track_operation_tool, get_context_report_tool]
)

# Execute with advanced context system
context = RunContext()
result = await agent(
    prompt="Initialize context for user 'john_doe' with session type 'analysis', update activity, set language to Spanish, track a 0.5s operation, and get context report",
    context=context
).collect()

# Comprehensive context data is available
print("Context Report:")
`}/>

The Run Context provides a powerful foundation for building complex, stateful AI applications with proper data sharing, configuration management, and performance tracking capabilities.
