---
title: Evals
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Automated Testing for Timbal Agents

---

AI outputs are non-deterministic — the same input can yield different results. **Evals** in Timbal help you measure, test, and improve your agents and flows with automated, LLM-powered checks.

## What Are Evals in Timbal?

Evals are automated tests that assess your agent's outputs and tool usage. They use LLMs to compare your agent's process (the tools it used and how) and its result (the answer it gave) to the expected process and result. Evals can be run locally, in the cloud, or as part of your CI/CD pipeline.

### Types of Evals

- <span style={{color: 'var(--timbal-purple)'}}><strong>Output Evals</strong></span>: Did the agent produce the correct answer, with proper content and formatting?
- <span style={{color: 'var(--timbal-purple)'}}><strong>Steps Evals</strong></span>: Did the agent use the right tools, in the right order, with the right inputs?
- <span style={{color: 'var(--timbal-purple)'}}><strong>Usage Evals</strong></span>: Did the agent's resource consumption (tokens, API calls) meet expectations or stay within expected bounds?

---

## How Evals Work

### Core Concepts

#### Test Suite
A collection of test cases defined in YAML format. A test suite can contain multiple tests that validate different aspects of your agent's behavior. Timbal automatically discovers all files matching the pattern `eval*.yaml` in your test directory.

#### Test
A single test case with multiple turns. Each test focuses on a specific scenario or functionality of your agent. Tests are defined within YAML files and can be run individually or as part of a suite.

#### Turn
One interaction between user and agent. A test can contain multiple turns to validate multi-step conversations. Each turn consists of:
- **Input**: What the user says or asks (text and optionally files) - **required**
- **Output**: What the agent should respond (validated against expected content) - *optional*
- **Steps**: Tools the agent should use (validated against expected tool calls) - *optional*
- **Usage**: Resource consumption limits (token counts) - *optional*

#### Validators
Programmatic checks that compare actual vs expected behavior. Validators use both exact matching and LLM-powered semantic evaluation to assess correctness.

---

## Test Structure

Evals are defined in YAML files with the following structure:

<CodeBlock language="bash" code={`- name: test_name
    description: Optional test description
    turns:
      - input:
          text: User input here
          files: ["path/to/file.pdf"]  # Optional file attachments
        output:
          text: Expected output text  # Optional: can use validators instead
          validators:
            contains:
              - "expected substring"
              - "another substring"
            not_contains:
              - "unwanted text"
            regex: "^Success.*"
            semantic: 
              - "Should provide helpful response about the topic"
        steps:
          validators:
            contains:
              - name: tool_name
                input:
                  parameter: expected_value
            not_contains:
              - name: unwanted_tool
            semantic: "Should use search tools to find information"
        usage:
          gpt-4.1:
            input_text_tokens:
              max: 5000
              min: 1000
            output_text_tokens:
              max: 2000
            input_text_tokens+output_text_tokens:  # Combined usage
              max: 6000`}/>

### Fields Reference

#### Test Level
- **`name`**: Unique identifier for the test (required)
- **`description`**: Human-readable description of what the test validates (optional)
- **`turns`**: Array of user-agent interactions to test (required)

#### Turn Level
- **`input`**: The user's message or query (required)
  - **`text`**: Text content of the message
  - **`files`**: Array of file paths to attach (optional)

  **Note**: You can use shorthand syntax: `input: "Your message"` is equivalent to `input: { text: "Your message" }`
- **`output`**: Expected agent response (optional)
  - **`text`**: Response text to store in conversation memory (optional)

  **Note**: You can use shorthand syntax: `output: "Response text"` is equivalent to `output: { text: "Response text" }`
- **`validators`**: Validation rules for the output (optional)
- **`steps`**: Expected tool usage (optional)
  - **`validators`**: Validation rules for tool calls (optional)
- **`usage`**: Resource consumption limits (optional)

---

## Validators

Timbal provides several types of validators for comprehensive testing:

### Output Validators

| Validator      | Description                                                    | Example Usage                           |
|---------------|----------------------------------------------------------------|-----------------------------------------|
| `contains`     | Checks if output includes specified substrings                | `contains: ["hello", "world"]`         |
| `not_contains` | Checks if output does NOT include specified substrings        | `not_contains: ["error", "failed"]`    |
| `regex`        | Checks if output matches a regular expression pattern          | `regex: "^Success: .+"`               |
| `semantic`     | Uses LLM to validate semantic correctness against reference    | `semantic: "Should greet user politely"` |

### Steps Validators

| Validator      | Description                                                    | Example Usage                           |
|---------------|----------------------------------------------------------------|-----------------------------------------|
| `contains`     | Checks if steps include specified tool calls with inputs      | `contains: [{"name": "search", "input": {"query": "test"}}]` |
| `not_contains` | Checks if steps do NOT include specified tool calls           | `not_contains: [{"name": "delete_file"}]` |
| `semantic`     | Uses LLM to validate tool usage against expected behavior     | `semantic: "Should search before providing answer"` |

### Usage Validators

Usage validators monitor resource consumption:

<CodeBlock language="bash" code={`usage:
    model_name:
      # OpenAI models exact token field names:
      input_text_tokens:
        max: 5000
        min: 100
      input_cached_tokens:
        max: 1000
      input_audio_tokens:
        max: 500
      output_text_tokens:
        max: 1000
      output_audio_tokens:
        max: 500
      output_reasoning_tokens:
        max: 2000
      
      # Anthropic models exact token field names:
      input_tokens:
        max: 5000
      output_tokens:
        max: 1000
      
      # Combined metrics using +
      input_text_tokens+output_text_tokens:
        max: 6000`}/>

---

## Running Evals

### Command Line Interface

<CodeBlock language="bash" code={`# Run all tests in a directory
python -m timbal.eval --fqn path/to/agent.py::agent_name --tests path/to/tests/

# Run a specific test file
python -m timbal.eval --fqn path/to/agent.py::agent_name --tests path/to/tests/eval_search.yaml

# Run a specific test by name
python -m timbal.eval --fqn path/to/agent.py::agent_name --tests path/to/tests/eval_search.yaml::test_basic_search`}/>

### Command Options

- **`--fqn`**: Fully qualified name of your agent (format: `file.py::agent_name`)
- **`--tests`**: Path to test file, directory, or specific test (format: `path/file.yaml::test_name`)

---

## Examples

### Example 1: Product Search with Complete Validation

<CodeBlock language="bash" code={`- name: eval_reference_product
  description: Test product search by reference code
  turns:
    - input:
        text: tell me the measurements of H214E/1
      steps: 
        validators:
          contains:
            - name: search_by_reference
              input:
                reference_code: H214E/1
          semantic: "Should search for product using the exact reference code H214E/1"
      output: 
        validators: 
          contains:
            - "H214E/1"
            - "95 mm"
            - "120 mm"
          not_contains:
            - "error"
            - "not found"
          semantic: "Should provide complete product measurements including diameter and height specifications"
      usage:
        - gpt-4.1:
            input_text_tokens:
              max: 5000
              min: 1000
            output_text_tokens:
              max: 2000`}/>

**How this test works:**
1. **Input**: User asks for measurements of product H214E/1
2. **Steps Validation**: 
   - `contains`: Verifies `search_by_reference` tool was called with correct parameters
   - `semantic`: Uses LLM to verify search logic was appropriate
3. **Output Validation**:
   - `contains`: Checks for specific measurement values and product code
   - `not_contains`: Ensures no error messages appear
   - `semantic`: Validates that response provides comprehensive product information
4. **Usage Validation**: Monitors token consumption within specified limits

### Example 2: Multi-turn Conversation with Memory

<CodeBlock language="bash" code={`- name: eval_memory_retention
  description: Test agent's ability to remember information across turns
  turns:
    - input: Hi, my name is David and I work in engineering
      output: Nice to meet you David! How can I help you?
    - input:
        text: What's my name and what do I do for work?
      steps:
        validators:
          not_contains:
            - name: search_external
          semantic: "Must not contain any tools"
      output:
        validators:
          contains: ["David", "engineering"]
          semantic: "Should recall both name and profession from previous turn"`}/>

**How multi-turn tests work:**
- **Turn 1**: Establishes context (input and output only) by providing information to remember - no validators needed
- **Turn 2**: Tests memory by asking for previously provided information - contains validators to verify behavior
- **Memory validation**: Ensures agent retrieves information from conversation history rather than external sources

### Example 3: File Processing with Error Handling

<CodeBlock language="bash" code={`- name: eval_file_processing
  description: Test file upload and processing with error scenarios
  turns:
    - input:
        text: Analyze this document for key metrics
        files: ["test_data/report.pdf"]
      steps:
        validators:
          contains:
            - name: extract_text_from_pdf
              input:
                file_path: report.pdf
            - name: analyze_metrics
          not_contains:
            - name: delete_file
          semantic: "Should extract text from PDF then analyze for metrics"
      output:
        validators:
          contains: ["metrics", "analysis"]
          not_contains: ["error", "failed", "unable"]
          regex: ".*\\d+.*%.*"  # Should contain percentage
          semantic: "Should provide quantitative analysis with specific metrics"`}/>

---

## Understanding Eval Results

Timbal generates comprehensive evaluation results in JSON format:

<CodeBlock language="bash" code={`{
    "total_tests": 5,
    "total_turns": 12,
    "outputs_passed": 10,
    "outputs_failed": 2,
    "steps_passed": 8,
    "steps_failed": 4,
    "usage_passed": 12,
    "usage_failed": 0,
    "execution_errors": 1,
    "tests_failed": [
      {
        "test_name": "eval_product_search",
        "test_path": "evals/product.yaml::eval_product_search",
        "input": {
          "text": "Find product X123",
          "files": []
        },
        "reason": ["steps", "output"],
        "execution_error": null,
        "output_passed": false,
        "output_explanations": [
          "Response did not include required product specifications"
        ],
        "actual_output": {
          "text": "I couldn't find that product.",
          "files": []
        },
        "expected_output": {
          "validators": {
            "semantic": ["Should provide product details and availability"]
          }
        },
        "steps_passed": false,
        "steps_explanations": [
          "No step found with tool 'search_product_catalog'."
        ],
        "actual_steps": [
          {
            "tool": "general_search",
            "input": {"query": "X123"}
          }
        ],
        "expected_steps": {
          "validators": {
            "contains": [{"name": "search_product_catalog"}]
          }
        },
        "usage_passed": true,
        "usage_explanations": []
      }
    ]
}`}/>

### Result Analysis

- **Summary metrics**: Total counts of passed/failed validations across all test types
- **Detailed failures**: Complete information about each failed test including:
  - **Actual vs Expected**: What the agent actually did vs what was expected
  - **Explanations**: Detailed reasons for failures from each validator
  - **Execution errors**: Runtime errors during test execution
- **Usage monitoring**: Resource consumption tracking for cost and performance optimization

---

## Summary

Timbal's evaluation framework provides:

- **Comprehensive Testing**: Validate outputs, tool usage, and resource consumption
- **Flexible Validation**: From exact string matching to semantic LLM-powered checks  
- **Multi-turn Support**: Test complex conversational flows and memory retention
- **Detailed Reporting**: Rich failure analysis for debugging and improvement
- **CI/CD Integration**: Automated testing to prevent regressions

This evaluation system helps you build reliable, testable AI agents that consistently produce correct results and follow expected processes, giving you confidence in your agent's behavior across different scenarios and edge cases.