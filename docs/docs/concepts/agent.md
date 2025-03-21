---
title: Agents
sidebar: 'docsSidebar'
---

# Agents

An agent is the join of having a `LLM` and `tools`. 

An agent is an LLM-powered component that can reason about and interact with provided tools to accomplish tasks. The agent will:
1. Receive an input prompt
2. Decide whether to use available tools or respond directly
3. If using tools, call them and incorporate their results
4. Continue this process until reaching a final response

```python
flow.add_agent(tools=[identity_handler])
```

Itself is the comination of doing:

```python
flow.add_step("tool", identity_handler)
flow.add_llm("llm_1", memory_id="llm")
flow.add_link("tool", "llm_1", is_tool=True)
flow.add_llm("llm_2", memory_id="llm")
flow.add_link("tool", "llm_2", is_tool_result=True)
```

As an LLM and tools the parameters you can choose are:

- `model`: The model to use. e.g. `gpt-4o-mini`
- `tools`: The tools to use. e.g.
You can pass the tools as:
  - A function or BaseStep directly e.g. `[identity_handler]`
  - A dict with keys:
    - `tool`: The function or BaseStep e.g. `identity_handler`
    - `description`: Optional description to help the agent understand the tool e.g. `"This is a tool that returns the input"`
- `name`: The name of the agent. Defaults to `agent`.
- `memory_id`: The memory id to use. e.g. `llm`. It will be the same memory for the first and second llm. Defaults to the agent name.
- `max_iter`: The maximum number of iterations the agent will run. Defaults to 1.
- `state_saver`: The state saver to use. Defaults to `None`.
- `kwargs`: Any other parameter to pass to the LLM.