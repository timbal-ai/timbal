# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing
- **Install dependencies**: `uv sync --dev` (uses uv package manager)
- **Run tests**: `pytest` or `uv run pytest` 
- **Run single test**: `pytest path/to/test_file.py::test_function`
- **Linting**: `ruff check` or `uv run ruff check`
- **Format code**: `ruff format` or `uv run ruff format`
- **Fix lint issues**: `ruff check --fix`

### Python Package Structure
- Main package: `python/timbal/`
- Tests: `python/tests/`
- Working directory for hatch: `python/`

## Architecture Overview

### Core Framework Structure
This is an AI agent framework called Timbal that provides two main execution patterns:

1. **Agent Pattern** (`python/timbal/core/agent/`): LLMs autonomously decide execution paths and tool usage
2. **Flow Pattern** (`python/timbal/core/flow/`): Explicit step-by-step workflow definition with data mapping

There's another directory in `python/timbal/core_v2` that is not being used yet. It is a refactoring of the core framework to make it more modular and easier to extend. It is in the works, do not focus on this for now unless instructed otherwise.

### Key Components

#### Core Engine (`python/timbal/core/`)
- `agent/engine.py`: Agent execution engine with autonomous tool selection
- `flow/engine.py`: Flow execution engine with explicit step orchestration  
- `base.py`: Base classes for all runnable components
- `step.py`: Individual step definitions
- `stream.py`: Event streaming and async handling

#### State Management (`python/timbal/state/`)
- `context.py`: Runtime execution context
- `data.py`: Data flow and mapping between steps
- `savers/`: Persistence layer (InMemory, JSONL, Platform)

#### LLM Integration (`python/timbal/steps/llms/`)
- `anthropic_llm.py`: Anthropic Claude integration
- `openai_llm.py`: OpenAI GPT integration  
- `gemini_llm.py`: Google Gemini integration
- `router.py`: Multi-provider routing logic

#### Tool Ecosystem (`python/timbal/steps/`)
- Modular tools organized by provider (elevenlabs, fal, gmail, etc.)
- Each tool directory contains focused functionality
- Tools auto-register with agents based on function signatures

#### Type System (`python/timbal/types/`)
- `message.py`: Unified message format across LLM providers
- `file.py`: File handling with automatic content detection
- `events/`: Event system for streaming responses
- Strong typing with Pydantic models throughout

### Data Flow Architecture
- **Agents**: Messages → Tool Selection → LLM → Tool Execution → Response
- **Flows**: Input → Step Chain → Data Mapping → Output
- Both patterns support streaming, state persistence, and event handling

### Testing Strategy
- Core tests in `python/tests/core/` mirror the main package structure
- Agent, flow, and component-specific test suites
- Async test support via pytest-asyncio

### Configuration
- `pyproject.toml`: Python packaging, dependencies, and tool config
- `timbal.yaml`: Project-specific configuration files in examples/sandbox
- Ruff for linting/formatting with 120 char line length