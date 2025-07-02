---
title: Evals
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Automated Testing for Timbal Agents

---

AI outputs are non-deterministic — the same input can yield different results. **Evals** in Timbal help you measure, test, and improve your agents and flows with automated, LLM-powered checks.

## What Are Evals in Timbal?

Evals are automated tests that assess your agent's outputs and tool usage. They use LLMs to compare your agent's process (the tools it used and how) and its result (the answer it gave) to the expected process and result. Evals can be run locally, in the cloud, or as part of your CI/CD pipeline

### Types of Evals

- <span style={{color: 'var(--timbal-purple)'}}><strong>Process Evals</strong></span>: Did the agent use the right tools, in the right order, with the right inputs?
- <span style={{color: 'var(--timbal-purple)'}}><strong>Result Evals</strong></span>: Did the agent produce the correct answer, regardless of wording or formatting?

---

## Getting Started

### 1. Prepare Your Test Cases

Create a YAML file (e.g. `evals.yaml`, `eval_response.yaml`) with the following structure:
<CodeBlock language="bash" code ={`- name: eval_reference_product
  description:
  turns:
    - input: dime las medidas de H214E/1
      steps: 
        contains: 
          - search_by_reference
        semantic: input reference code should be 'H214E/1'
      output: 
        validators: 
          semantic: Las medidas de la Pilona Hospitalet Extraíble Ref. H214E/1 son *Diámetro Ø A 95 mm *B 120 mm *Altura total H 970 mm *Altura extraíble H1 170 mm Puedes ver más detalles y descargar ficha técnica, instrucciones de montaje, etc. en el siguiente enlace [Ver producto Pilona Hospitalet Extraíble H214E/1](https://www.benito.com/es/movilidad/bolardos-pilonas/hospitalet-extraible--H214E.html) [Ficha técnica](https://www.benito.com/es/download/BENITO_data-sheet-H214E) [Especificaciones técnicas (CAD)](https://www.benito.com/es/download/BENITO_H214E_FP.dwg) [Instrucciones de montaje](https://www.benito.com/es/download/BENITO_H214E-MU) [Plan de mantenimiento](https://www.benito.com/es/download/BENITO_H214E_HM.pdf)
      usage:
        - gpt-4.1:
            input_text_tokens:
              max: 5000
              min: 1000
            output_text_tokens:
              max: 5000
`}/>
### 2. Set Up Your Evaluation Agents

Define evaluation prompts for process and result checking, and create Timbal agents for each:

<CodeBlock language="bash" code ={`from timbal import Agent

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


---

## Understanding Eval Results

Each eval returns a json:
<CodeBlock language="bash" code ={`{
  "total_tests": 1,
  "total_turns": 32,
  "outputs_passed": 26,
  "outputs_failed": 6,
  "steps_passed": 14,
  "steps_failed": 6,
  "usage_passed": 0,
  "usage_failed": 0,
  "tests_failed": [
    {
      "test_path": "evals.yaml::eval_problemas_pantalla_foto",
      "input": {
        "text": "Buen dia equipo. Me echan una mano por favor, ya la he encendido para de veces y he apagado el ordenador para de veces también y se me queda así la pantalla, tengo el simulador colocado en temas de conexión está todo ok",
        "files": [
          "data:image/png;base64,iVBORw0KGgoAAAANSUh...
        ]
      },
      "reason": [
        "steps",
        "output"
      ],
      "output_passed": false,
      "output_explanations": [
        "La respuesta proporcionada no aborda el problema expuesto en la referencia, que se trata de un error relacionado con la forma de encender el ordenador. En su lugar, la respuesta pide especificación del problema con la 'pantalla', lo cual no es relevante ni útil en este contexto."
      ],
      "actual_output": {
        "text": "¿Puedes especificar exactamente qué problema estás teniendo con la pantalla?",
        "files": []
      },
      "expected_output": {
        "validators": {
          "semantic": [
            "¡hola! Gracias por toda la información. Este error ocurre cuando mantienes pulsado el botón de encendido al encender el ordenador. Prueba a apagarlo y volver a encenderlo, pulsando una única vez el botón."
          ]
        }
      },
      "steps_passed": false,
      "steps_explanations": [
        "No step found with tool 'search_info'."
      ],
      "actual_steps": [],
      "expected_steps": {
        "validators": {
          "contains_steps": [
            {
              "name": "search_info"
            }
          ]
        }
      },
      "usage_passed": true,
      "usage_explanations": []
    },
`}/>

You can use these results to:
- Track your agent's quality over time
- Debug and improve your prompts, tools, or workflows
- Catch regressions before they reach production

---

## Summary

- **Evals** in Timbal let you automatically check if your agent is doing the right thing, both in process and result.
- You get detailed, LLM-powered feedback for every test case.
- This helps you build more reliable, trustworthy AI systems.