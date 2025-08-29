---
title: Advanced Workflow Concepts
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Advanced Workflow Concepts

:::warning[Work in Progress]
The Workflow class is currently under development and not fully implemented. The examples in this documentation may not work as expected.
:::

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Build complex workflows with nested components, tool integration, and advanced parameter mapping.
</h2>

---

## Nested Workflows

You can use a workflow as a step inside another workflow, enabling modular, reusable workflow components.

### Example: Modular Workflow Components

Suppose you want to process incoming orders by first validating them and then confirming them. You can define a reusable subworkflow for validation and use it in your main workflow.

<CodeBlock language="python" code={`from timbal.core import Workflow

def validate_order(order: dict) -> dict:
    # Simulate checking if the order is valid
    if order["quantity"] > 0 and order["item"] in ["apple", "banana"]:
        return {"valid": True, "item": order["item"], "quantity": order["quantity"]}
    else:
        return {"valid": False, "reason": "Invalid item or quantity"}

def confirm_order(validated: dict) -> str:
    if validated["valid"]:
        return f"Order confirmed: {validated['quantity']} {validated['item']}(s)."
    else:
        return f"Order failed: {validated['reason']}"

# Define the reusable subworkflow for validation
validation_workflow = (
    Workflow(name="validation")
    .add_step(validate_order)
)

# Main workflow uses the subworkflow and then confirms the order
main_workflow = (
    Workflow(name="order_processor")
    .add_step(lambda: {"item": "apple", "quantity": 3})
    .add_step(validation_workflow)
    .add_step(confirm_order)
)

async def main():
    result = await main_workflow.complete()
    print(result.output)  # Will contain the final result from confirm_order
`}/>

---

## Tool Integration

Workflows can integrate with tools (functions or APIs) as part of their execution pipeline. Tools are automatically converted to steps when added to workflows.

### Example: Workflow with External API Tools

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool

def get_weather(city: str) -> str:
    # Simulate a weather API call 
    return f"The weather in {city} is sunny and 25°C."

def format_weather_report(weather: str, city: str) -> str:
    return f"Weather Report for {city}: {weather}"

weather_tool = Tool(handler=get_weather)

workflow = (
    Workflow(name="weather_reporter")
    .add_step(weather_tool, city="Paris")
    .add_step(format_weather_report, city="Paris")
)

async def main():
    result = await workflow.complete()
    print(result.output)  # Will contain the formatted weather report
`}/>

---

## Advanced Data Flow

Workflows support sophisticated data flow between steps, allowing you to control exactly how data moves through your pipeline.

### Multiple Input Parameters

<CodeBlock language="python" code={`def get_user_info() -> dict:
    return {"name": "Alice", "age": 30, "city": "New York"}

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 25°C."

def create_report(user_info: dict, weather: str) -> str:
    return f"Report for {user_info['name']} ({user_info['age']} years old): {weather}"

workflow = (
    Workflow(name="user_weather_report")
    .add_step(get_user_info)
    .add_step(get_weather, city="New York")  # Fixed city parameter
    .add_step(create_report)  # Receives both user_info and weather
)`}/>

### Conditional Data Flow

You can use conditional logic to determine which data to pass:

<CodeBlock language="python" code={`def check_user_type(user_info: dict) -> str:
    return "vip" if user_info.get("is_vip") else "regular"

def get_vip_weather(city: str) -> str:
    return f"VIP weather report for {city}: Sunny and 25°C with premium forecast."

def get_regular_weather(city: str) -> str:
    return f"Weather in {city}: Sunny and 25°C."

def create_weather_report(user_type: str, weather: str) -> str:
    return f"{weather} (User type: {user_type})"

workflow = (
    Workflow(name="conditional_weather")
    .add_step(lambda: {"name": "Alice", "is_vip": True, "city": "Paris"})
    .add_step(check_user_type)
    .add_step(get_vip_weather, city="Paris")  # This will be used for VIP users
    .add_step(create_weather_report)
)`}/>

---

## Error Handling and Debugging

Workflows provide built-in error handling and debugging capabilities through event streaming.

### Streaming Events for Debugging

<CodeBlock language="python" code={`async def debug_workflow():
    workflow = (
        Workflow(name="debug_example")
        .add_step(lambda: {"data": "test"})
        .add_step(lambda x: x["data"].upper())
    )
    
    async for event in workflow.run():
        print(f"Event: {event.type} - {event.path}")
        if hasattr(event, 'output'):
            print(f"Output: {event.output}")
        if hasattr(event, 'error'):
            print(f"Error: {event.error}")
`}/>

### Error Recovery

Workflows can handle errors gracefully and continue execution:

<CodeBlock language="python" code={`def risky_operation(data: dict) -> str:
    if data.get("should_fail"):
        raise ValueError("Operation failed")
    return "Operation successful"

def fallback_operation(data: dict) -> str:
    return "Fallback operation completed"

workflow = (
    Workflow(name="error_handling")
    .add_step(lambda: {"should_fail": False, "data": "test"})
    .add_step(risky_operation)
    .add_step(fallback_operation)  # This will run even if risky_operation fails
)`}/>

---

## Performance Optimization

Workflows support various optimization techniques for better performance.

### Parallel Execution

While workflows execute steps sequentially by default, you can optimize performance by structuring your workflow efficiently:

<CodeBlock language="python" code={`def fetch_user_data(user_id: str) -> dict:
    # Simulate API call
    return {"id": user_id, "name": "Alice"}

def fetch_order_data(user_id: str) -> dict:
    # Simulate API call
    return {"user_id": user_id, "orders": [1, 2, 3]}

def combine_data(user_data: dict, order_data: dict) -> dict:
    return {
        "user": user_data,
        "orders": order_data["orders"]
    }

# This workflow fetches data sequentially but efficiently
workflow = (
    Workflow(name="data_combiner")
    .add_step(fetch_user_data, user_id="123")
    .add_step(fetch_order_data, user_id="123")
    .add_step(combine_data)
)`}/>

---

## State Management

Workflows can maintain state across executions using the state management system.

### Persistent State

<CodeBlock language="python" code={`from timbal.state import get_run_context

def counter_step() -> int:
    context = get_run_context()
    current_count = context.get("counter", 0)
    new_count = current_count + 1
    context["counter"] = new_count
    return new_count

workflow = (
    Workflow(name="stateful_counter")
    .add_step(counter_step)
    .add_step(counter_step)
    .add_step(counter_step)
)

# Each execution will increment the counter
async def main():
    result1 = await workflow.complete()  # Counter: 1, 2, 3
    result2 = await workflow.complete()  # Counter: 4, 5, 6
`}/>

---

For more, see the [Workflows Overview](/workflows) and [Examples](/examples).
