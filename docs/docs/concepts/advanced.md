---
title: Advanced
sidebar: 'docsSidebar'
---

# Advanced Timbal Concepts

Welcome to the advanced concepts guide! Here you'll learn about the powerful features that make Timbal truly flexible and powerful. Let's dive in!

## Steps: The Building Blocks

Steps are the fundamental units of your flow. Think of them as Lego blocks that you can combine in creative ways!

### 1. Function-Based Steps

The simplest way to create steps is using Python functions:

```python
def identity_handler(x: Any = Field(default=None)) -> Any:
    return x
```

:::tip[Pro Tip]
Using `Field` is highly recommended because it:
- Validates inputs and outputs automatically
- Provides clear documentation for tool integration
- Enables beautiful platform interfaces
:::

### 2. Built-in Timbal Steps

Timbal provides specialized steps for common use cases:

```python
step = Step(handler_fn=identity_handler)
```

### 3. Custom Steps

Create your own step classes by inheriting from `BaseStep`:

```python
from timbal import BaseStep, BaseModel
from typing import Any

class MyCustomStep(BaseStep):
    def params_model(self) -> BaseModel:
        # Define input parameters schema
        pass
    def params_model_schema(self) -> dict[str, Any]:
        # Return JSON schema for parameters
        pass
    def return_model(self) -> BaseModel:
        # Define return value schema
        pass
    def return_model_schema(self) -> dict[str, Any]:
        # Return JSON schema for return value
        pass
    def run(self, kwargs: Any) -> Any:
        # Implement step logic
        pass
```

## 🔗 Conditional Flow Control

Want to make your flows smarter? Use conditional links!

```python
flow.add_link(
    "step_1", 
    "step_2", 
    condition="step_1.return == 'Hello'") # Magic condition!
```

### Advanced Conditions

```python
# Multiple conditions
flow.add_link(
    "step_1",
    "step_2",
    condition="step_1.return > 10 and step_1.status == 'success'"
)

# Using functions
flow.add_link(
    "step_1",
    "step_2",
    condition=lambda x: x > 10
)
```

## Data Value Magic

Combine and transform data between steps with template syntax:

```python
flow = (
    Flow()
    .add_step("step_1", identity_handler, x=1)
    .add_step("step_2", identity_handler, x=2)
    .add_step("step_3", identity_handler)
    .set_data_value("step_3.x", "{{step_1.return}} {{step_2.return}}") # Input: "1 2"
    .set_output("step_3_x", "step_3.x")
    .set_output("step_3_return", "step_3.return")
)
```

### Template Features

1. **Basic Interpolation**
   ```python
   "Hello {{name}}!"  # Simple variable
   ```

2. **Math Operations**
   ```python
   "{{x + y}}"  # Addition
   "{{x * y}}"  # Multiplication
   ```

3. **String Operations**
   ```python
   "{{text.upper()}}"  # Uppercase
   "{{text.split()}}"  # Split string
   ```

## Advanced Flow Patterns

### 1. Parallel Processing
```python
flow = (
    Flow()
    .add_step("step_1", process_data)
    .add_step("step_2", process_data)
    .add_step("step_3", combine_results)
    .add_link("step_1", "step_3")
    .add_link("step_2", "step_3")
)
```

### 2. Error Handling
```python
flow = (
    Flow()
    .add_step("main_step", process_data)
    .add_step("error_handler", handle_error)
    .add_link("main_step", "error_handler", condition="main_step.status == 'error'")
)
```

### 3. Looping
```python
flow = (
    Flow()
    .add_step("start", initialize)
    .add_step("process", process_item)
    .add_step("check", check_condition)
    .add_link("start", "process")
    .add_link("process", "check")
    .add_link("check", "process", condition="check.continue_loop")
)
```

## Best Practices

1. **Step Design**
   - Keep steps focused and single-purpose
   - Use meaningful names
   - Document inputs and outputs

2. **Flow Organization**
   - Group related steps
   - Use clear naming conventions
   - Document complex flows

3. **Error Handling**
   - Always handle potential errors
   - Use appropriate error messages
   - Implement fallback strategies

4. **Performance**
   - Optimize step execution
   - Use parallel processing when possible
   - Monitor resource usage

## Next Steps

Ready to take your Timbal skills to the next level? Check out:
- [Flows](flows.md): Master flow creation
- [Tools](tools.md): Extend functionality
