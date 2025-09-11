---
title: Conditional Branching
sidebar: 'examples'
draft: True
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Workflows often need to follow different paths based on a condition. These examples demonstrate how to create conditional flows using Timbal's workflow orchestration capabilities.

## Conditional logic using steps

In this example, the workflow uses conditional logic to execute one of two steps based on a condition. If the input value is less than or equal to 10, it runs `less_than_step` and returns 0. If the value is greater than 10, it runs `greater_than_step` and returns 20.

<CodeBlock language="python" code={`
`}/>

## Example usage

<CodeBlock language="python" code={`
`}/>