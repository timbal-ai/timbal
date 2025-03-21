---
title: Links
sidebar: 'docsSidebar'
---

# Links


A Link represents a connection between two steps in a workflow graph.

Links can be conditional (using a condition string that evaluates against workflow data), or represent tool calls and their results. 

Each link connects a source step (`step_id`) to a destination step (`next_step_id`).

## Attributes

| Attribute                               | Parameter                | Type                                | Description                                                                                                          |
| :-------------------------------------- | :----------------------- | :--------------------------------------- | :------------------------------------------------- |
| **Step ID**                                | `step_id`                   | `str`                         | The ID of the source step.                                                           |
| **Next Step ID**                                | `next_step_id`                   | `str`                         | The ID of the destination step.                                                           |
| **Condition** _(Optional)_                                | `condition`                   | `str`                         | The condition to evaluate.                                                           |
| **Tool**                                | `is_tool`                   | `bool`                         | Whether the link represents a tool call. Default to False                                                           |
| **Tool Result**                                | `is_tool_result`                   | `bool`                         | Whether the link represents a tool result. Default to False.                                                           |
| **Metadata**                           | `metadata`              | `dict[str, Any]`                         | Additional metadata associated with the link.                                                                 |