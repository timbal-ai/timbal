---
title: Steps
sidebar: 'docsSidebar'
draft: true
---

# Steps

Steps are the building blocks of your flow.

A step can be a function or a `BaseStep`. There are three main ways to create steps in Timbal:

1. **Using Functions**
   - Any Python function can be used as a step
   - Function parameters and return types are automatically validated

   ```python
    def identity_handler(x: Any = Field(default=None)) -> Any:
        return x
   ```

   :::note[Best Practice 💡]
   Using the `Field` class is recommended because it:
   - Validates inputs and outputs automatically
   - Provides a clear description of the inputs. Useful for tool integration.
   - Enables integration with platform interfaces
   :::

2. **Using Built-in Timbal Steps**
   - Timbal provides specialized step named `Step`
   - These steps have additional functionality for their specific use cases

   ```python
   step = Step(handler_fn=identity_handler)
   ```

3. **Creating Custom Steps**
   - You can create your own step classes by inheriting from `BaseStep`
   - Custom steps must implement the required abstract methods

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