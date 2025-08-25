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

1. **Agent Pattern** (`python/timbal/core/agent.py`): LLMs autonomously decide execution paths and tool usage
2. **Workflow Pattern** (`python/timbal/core/workflow.py`): Explicit step-by-step workflow definition with data mapping (early work)

### Key Components

#### Core Engine (`python/timbal/core/`)
- `runnable.py`: Base class for all executable components with event streaming, tracing, and async execution
- `agent.py`: Agent execution engine with autonomous tool selection and LLM orchestration
- `workflow.py`: Workflow execution engine with explicit step orchestration (early development stage)
- `tool.py`: Tool wrapper for function-based components
- `llm_router.py`: Multi-provider LLM routing logic

#### State Management (`python/timbal/state/`)
- `context.py`: Runtime execution context and state persistence
- `tracing/`: Execution tracing and monitoring system

#### Handler Ecosystem (`python/timbal/handlers/`)
- Modular handlers organized by provider (elevenlabs, fal, gmail, etc.)
- Each handler directory contains focused functionality
- Handlers auto-register with agents based on function signatures

#### Type System (`python/timbal/types/`)
- `message.py`: Unified message format across LLM providers
- `file.py`: File handling with automatic content detection
- `events/`: Event system for streaming responses
- Strong typing with Pydantic models throughout

#### Collectors (`python/timbal/collectors/`)
- Output processing and collection system
- Provider-specific collectors (anthropic, openai, etc.)
- Registry system for automatic collector selection

### Data Flow Architecture
- **Agents**: Messages → Tool Selection → LLM → Tool Execution → Response
- **Workflows**: Input → Step Chain → Data Mapping → Output (early stage)
- Both patterns support streaming, state persistence, and event handling via the Runnable base class

### Testing Strategy
- Core tests in `python/tests/core/` mirror the main package structure
- Agent, flow, and component-specific test suites
- Async test support via pytest-asyncio

### Configuration
- `pyproject.toml`: Python packaging, dependencies, and tool config
- Ruff for linting/formatting with 120 char line length