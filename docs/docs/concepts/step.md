---
title: Steps
sidebar: 'docsSidebar'
---

# Steps

Steps are the fundamental processing units within a workflow. They represent individual operations, from simple functions to complex language models, that can be connected together to create sophisticated workflows.

## Overview

The Step class encapsulates a handler function and provides functionality to:
    - Validate input parameters using Pydantic models
    - Validate return values using Pydantic models
    - Execute the handler function with proper parameter passing
    - Support both synchronous and asynchronous execution


## Attributes

| Attribute                               | Parameter                | Type                          | Description                                                                                                          |
| :-------------------------------------- | :----------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Function**                                | `handler_fn`                   | `str`                         | The function that implements the step's processing logic.                                                           |
| **Input Parameters Model**                                | `handler_params_model`                   | `str`                         | The Pydantic model for validating input parameters.                                                           |
| **Output Parameters Model**                           | `handler_return_model`              | `str`                         | The Pydantic model for validating return values.                                                                 |
| **LLM**                    | `is_llm`                    | `bool`                         | Whether the step is a LLM.                                                                                        |
| **Async**                    | `is_coroutine`                    | `bool`                         | Whether the step is a coroutine.                                                                                        |
| **Async Generator**                    | `is_async_gen`                    | `bool`                         | Whether the step is an async generator.                                                                                        |

