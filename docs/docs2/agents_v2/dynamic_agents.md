---
title: Dynamic Agents
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Dynamic Agents
<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Learn how to create agents with dynamic system prompts that update automatically using real time data.
</h2>

---

## Dynamic System Prompts

Agents support dynamic system prompts that can include live data through template functions. These functions are executed each time the agent runs, providing fresh context.

Use `{module::function}` syntax to embed dynamic values:

<CodeBlock language="python" code={`agent = Agent(
    name="dynamic_agent",
    model="openai/gpt-4o-mini",
    system_prompt="""You are a time-aware assistant.
    Current time: {datetime::datetime.now}."""
)`}/>

The previous example used a built-in function (datetime). You can also create your own custom functions:

<CodeBlock language="python" code={`# my_functions.py
def get_server_status():
    """Get server status."""
    status = check_server()  # Calls external function
    return f"Server: {status}"

agent = Agent(
    name="custom_agent", 
    model="openai/gpt-4o-mini",
    system_prompt="""You are a helpful assistant.
    Status: {my_functions::get_server_status}."""
)`}/>

:::warning
This feature is currently in development.
:::

You can also pass dynamic parameters to these functions using `RunContext` data that you previously set in the context.

<CodeBlock language="python" code={`# my_functions.py
from timbal.state import get_run_context

def get_user_language():
    """Get user language from context."""
    context = get_run_context()
    return context.data.get("user_lang", "english")

# Set context data first
context = RunContext(data={"user_lang": "spanish"})
set_run_context(context)

agent = Agent(
    name="multilang_agent",
    model="openai/gpt-4o-mini",
    system_prompt="Respond in {my_functions::get_user_language}. You are a helpful assistant."
)
await agent(prompt="Which is the capital of Germany?").collect()`}/>

The response will be in Spanish. You can change the language by simply updating the RunContext data:

<CodeBlock language="python" code={`# Change language to Catalan
context = RunContext(data={"user_lang": "catalan"})
set_run_context(context)

await agent(prompt="Which is the capital of Germany?").collect()
# Now the response will be in Catalan`}/>

The framework supports both synchronous and asynchronous functions automatically.

**Benefits:**

<div className="timeline">
<div className="timeline-item">
<div className="timeline-content">

<h4>Real-time context</h4>
System prompts reflect current state

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Dynamic behavior</h4>
Agent adapts to changing conditions

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Automatic execution</h4>
Functions run on each conversation

</div>
</div>

<div className="timeline-item">
<div className="timeline-content">

<h4>Performance</h4>
<h3></h3>
Template resolution is fast and cached

</div>
</div>
</div>

<style>{`
.timeline {
  display: flex;
  align-items: center;
  margin: 1rem 0;
  overflow-x: auto;
  padding: 0.5rem;
}

.timeline-item {
  width: 180px;
  text-align: left;
  padding: 0.5rem;
  background: var(--ifm-background-color);
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.timeline-connector {
  color: var(--ifm-color-primary);
  font-size: 1.2rem;
  padding: 0 0.5rem;
}

.timeline-content {
  padding: 0.5rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.timeline-content h3 {
  margin-bottom: 0.25rem;
  font-size: 1rem;
}

.timeline-content ul {
  list-style: disc;
  padding-left: 1.2em;
  margin: 0;
}

.timeline-content li {
  margin: 0.15rem 0;
  font-size: 0.85rem;
}

.timeline-content h4 {
  color: var(--ifm-color-primary);
  font-weight: bold;
  margin-bottom: 0.5em;
  margin-top: 0;
}
`}</style>