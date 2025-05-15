---
title: Evals
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Automated Testing for Timbal Agents

AI outputs are non-deterministic — the same input can yield different results. **Evals** in Timbal help you measure, test, and improve your agents and flows with automated, LLM-powered checks.

---

## What Are Evals in Timbal?

Evals are automated tests that assess your agent's outputs and tool usage. They use LLMs to compare your agent's process (the tools it used and how) and its result (the answer it gave) to the expected process and result. Evals can be run locally, in the cloud, or as part of your CI/CD pipeline

### Types of Evals

- <span style={{color: 'var(--timbal-purple)'}}><strong>Process Evals</strong></span>: Did the agent use the right tools, in the right order, with the right inputs?
- <span style={{color: 'var(--timbal-purple)'}}><strong>Result Evals</strong></span>: Did the agent produce the correct answer, regardless of wording or formatting?

---

## Getting Started

### 1. Prepare Your Test Cases

Create a CSV file (e.g. `eval.csv`) with columns like:
- `prompt`: The question or task for the agent.
- `result`: The expected answer.
- `process`: The expected sequence of tools and inputs.

### 2. Set Up Your Evaluation Agents

Define evaluation prompts for process and result checking, and create Timbal agents for each:

<CodeBlock language="python" code ={`from timbal import Agent

EVAL_PROCESS_PROMPT = """You are a helpful assistant that evaluates if the process done by the llm is correct.
Compare two processes: the first is correct, check if in the second were used the correct tools and the correct inputs.
Ignore case and formatting differences in all inputs.
If the process is correct, return true, otherwise return false.
Return: {"explanation": "Brief reasoning", "correct": true/false}
"""

EVAL_RESULT_PROMPT = """You are a helpful assistant that evaluates if the result is correct.
Compare two results: the first is correct, check if the second answers the question.
You have to focus on the answer being right (dimensions, material, codes, links, lists, etc.) not on the exact wording.
If the result is correct, return true, otherwise return false.
Return: {"explanation": "Brief reasoning", "correct": true/false}
"""

eval_process = Agent(model="gpt-4.1-nano", system_prompt=EVAL_PROCESS_PROMPT)
eval_result = Agent(model="gpt-4.1-nano", system_prompt=EVAL_RESULT_PROMPT)`}/>

### 3. Run the Evals

For each test case:
- Run your agent or flow with the prompt.
- Collect the output and the tools used (from the agent's memory).
- Pass the actual and expected process/result to the evaluation agents.
- Get a pass/fail and an explanation for each.

<CodeBlock language="python" code ={`import pandas as pd
from timbal.state import RunContext
# Your Timbal flow
from flow import flow  

csv_eval = pd.read_csv("./eval.csv")
for index, row in csv_eval.iterrows():
    prompt = row["prompt"]
    expected_result = row["result"]
    expected_process = row["process"]

    flow_output_event = await flow.complete(prompt=prompt)
    result = flow_output_event.output.content[0].text
    run_context = RunContext(parent_id=flow_output_event.run_id)

    # Extract tools_used from memory (see your Timbal memory docs for details)
    # Your logic here
    tools_used = ...  

    # Evaluate process
    eval_process_prompt = f"Question: {prompt}\nProcess: {tools_used}\nExpected process: {expected_process}"
    eval_process_output = await eval_process.complete(context=run_context, prompt=eval_process_prompt)
    print(eval_process_output.output.content[0].text)

    # Evaluate result
    eval_result_prompt = f"Question: {prompt}\nResult: {result}\nExpected result: {expected_result}"
    eval_result_output = await eval_result.complete(context=run_context, prompt=eval_result_prompt)
    print(eval_result_output.output.content[0].text)`}/>

---

## Understanding Eval Results

Each eval returns:
- **correct**: true/false
- **explanation**: a brief reasoning for the decision

You can use these results to:
- Track your agent's quality over time
- Debug and improve your prompts, tools, or workflows
- Catch regressions before they reach production

---

## Summary

- **Evals** in Timbal let you automatically check if your agent is doing the right thing, both in process and result.
- You get detailed, LLM-powered feedback for every test case.
- This helps you build more reliable, trustworthy AI systems.