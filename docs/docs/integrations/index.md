---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Timbal Integrations

Timbal supports a variety of integrations to extend its capabilities with external services such as speech, language, and data tools. This page provides general guidance for using integrations in your projects.

---

## Package Installation

Some integrations require additional Python packages. To simplify setup, you can install all necessary dependencies for a specific integration using:

<CodeBlock language="bash" code ={`uv add timbal[<integration>]`}/>

Replace `<integration>` with the name of the integration you want to use (e.g., `elevenlabs`, `gmail`, etc).

## API Keys & Environment Configuration

Many integrations require an API key for authentication. If so:

1. **Obtain an API key** from the service provider (see the integration's documentation for details).
2. **Set the API key as an environment variable** so Timbal can access it. For example:

   <CodeBlock language="bash" code ={`export SERVICE_API_KEY='your-api-key-here'`}/>
   Replace `SERVICE_API_KEY` with the variable name required by the integration (e.g., `ELEVENLABS_API_KEY`, `GMAIL_API_KEY`).

## Next Steps

- See the documentation for each integration for specific setup and usage instructions.
- If you encounter issues, check that all required packages are installed and API keys are correctly configured.