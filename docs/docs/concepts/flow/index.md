---
title: Flows
sidebar: 'docsSidebar'
---

import DocCardList from '@theme/DocCardList';
import { useCurrentSidebarCategory } from '@docusaurus/theme-common';


# Flows

A `Flow` is the core building block of the Timbal framework.

It represents a directed acyclic graph (DAG) where nodes are `steps` and edges define the execution order and data dependencies.

Flows are designed to orchestrate complex workflows by connecting different types of steps, managing data flow between them, and handling state persistence.

## Overview

Flows allow you to:
- Connect multiple steps in a directed, acyclic manner
- Pass data between steps using explicit mappings
- Handle both synchronous and asynchronous execution
- Integrate LLM-powered components
- Create reusable workflow components
- Stream results in real-time
- Persist state between executions

## Basic Structure

A Flow consists of:

<DocCardList items={[
  { type: 'link', label: 'Steps', href: '/docs/concepts/step', description: 'Individual processing units: functions, LLMs, subflows, custom step implementations' }, 
  { type: 'link', label: 'Links', href: '/docs/concepts/link', description: 'Connections between steps that define: Execution order, Data dependencies, Conditional execution,Tool relationships for LLM agents' },
]}/>

## Attributes

| Attribute                               | Parameter                | Type                                | Description                                                                                                          |
| :-------------------------------------- | :----------------------- | :--------------------------------------- | :------------------------------------------------- |
| **Steps**                                | `steps`                   | `dict[str, BaseStep]`                         | A dictionary mapping step IDs to step instances.                                                           |
| **Links**                                | `links`                   | `dict[str, Link]`                         | A dictionary mapping link IDs to link instances.                                                           |
| **Data**                                | `data`                   | `dict[str, Any]`                         | A dictionary storing flow data and mappings.                                                           |
| **Outputs**                                | `outputs`                   | `dict[str, str]`                         | A dictionary mapping output names to data keys.                                                           |
| **State Saver** _(Optional)_                                | `state_saver`                   | `Optional[BaseSaver]`                         | An optional saver for persisting flow state.                                                           |
| **LLM**                                | `is_llm`                   | `bool`                         | Whether this flow acts as an LLM. Always False.                                                           |
| **Coroutine**                                | `is_coroutine`                   | `bool`                         | Whether this flow returns a coroutine.                                                           |
| **Async Generator**                                | `is_async_gen`                   | `bool`                         | Whether this flow returns an async generator.                                                           | 
| **Compiled**                                | `_is_compiled`                   | `bool`                         | Whether the flow has been compiled.                                                           |
