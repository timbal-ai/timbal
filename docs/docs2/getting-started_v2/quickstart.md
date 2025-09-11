---
sidebar_position: 3
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Quickstart

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Build your first AI application with Timbal in just 5 lines of code.
</h2>

---

We'll start implementing an <span style={{color: 'var(--timbal-purple)'}}><strong>agent</strong></span>. It will be a <span style={{color: 'var(--timbal-purple)'}}><strong>simple chatbot</strong></span> and gradually enhance it with advanced features. Let's dive in!

Before moving forward, ensure you've completed the installation of Timbal. If you haven't set it up yet, follow the [installation guide](/getting-started/installation) to get started.



## Part 0: Creating a Timbal Project

Timbal provides a CLI tool that automatically sets up your project structure and dependencies.

When you run `timbal init`, it 
creates a directory with the following structure:

<div class="vertical-stepper">
  <div class="step">
    <div class="circle">1</div>
    <div class="step-content">
      <div class="step-title">Create Project Structure</div>
      <div class="step-desc">
      Run the Timbal CLI command to create a new project:

        <CodeBlock language="bash" code ={`timbal init my-project`}/>

This creates a complete project structure with all necessary files and dependencies.
      </div>
    </div>
  </div>
  <div class="step">
    <div class="circle">2</div>
    <div class="step-content">
      <div class="step-title">Project Structure</div>
      <div class="step-desc">
      Your project will have this structure:
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '2rem',
            flexWrap: 'wrap',
            maxWidth: '900px',
            margin: '1.5rem auto',
            
          }}
        >
          <div className="file-tree-box" style={{ fontSize: '0.85em', maxWidth: '320px' }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.2rem' }}>
              <img src="/img/folder.svg" className="file-tree-icon" style={{ width: '14px', marginRight: '0.5rem' }} />
              <span>my-project</span>
            </div>
            <div className="file-list" style={{ marginLeft: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.2rem' }}>
                <img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px', marginRight: '0.5rem' }} />
                <span>agent.py</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.2rem' }}>
                <img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px', marginRight: '0.5rem' }} />
                <span>.dockerignore</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.2rem' }}>
                <img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px', marginRight: '0.5rem' }} />
                <span>pyproject.toml</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px', marginRight: '0.5rem' }} />
                <span>timbal.yaml</span>
              </div>
            </div>
          </div>
          <div style={{ minWidth: '320px', flex: 1 }}>
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Purpose</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><b>agent.py</b></td>
                  <td>Main application file where you define your AI agents and workflows</td>
                </tr>
                <tr>
                  <td><b>.dockerignore</b></td>
                  <td>Docker configuration for containerized deployment</td>
                </tr>
                <tr>
                  <td><b>pyproject.toml</b></td>
                  <td>Python project configuration and dependency management</td>
                </tr>
                <tr>
                  <td><b>timbal.yaml</b></td>
                  <td>Timbal framework configuration and settings</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="step">
    <div class="circle">3</div>
    <div class="step-content">
      <div class="step-title">Personalize Your Project</div>
      <div class="step-desc">
      You will have a file `agent.py` with a basic structure of a flow.
This is the main file you'll edit to build your agent.

<CodeBlock language="python" title= "agent.py" code ={`from datetime import datetime

from timbal import Agent
from timbal.handlers.docx import create_docx

def get_datetime() -> str:
    return datetime.now().isoformat()

agent = Agent(
    name="demo_agent",
    model="openai/gpt-5-mini",
    tools=[get_datetime, create_docx]
)

await agent(prompt="What time is it?").collect()
await agent(prompt="Cool, write that down on a word file for me").collect()`}/>

👀 If you want to use LLM models, you might have to provide your API keys for the LLM providers corresponding to the models you are using.

:::tip[Security Note]
Store sensitive information like API keys in a `.env` file and never commit it to version control.
:::
</div>
    </div>
  </div>
  <div class="step">
    <div class="circle">4</div>
    <div class="step-content">
      <div class="step-title"> Run Your Flow</div>
      <div class="step-desc">
      Run your agent using Python:

<CodeBlock language="bash" code ={`python agent.py`}/>

Or with uv:

<CodeBlock language="bash" code ={`uv run agent.py`}/>

Your agent will start and be ready to receive prompts!
      </div>
    </div>
  </div>
</div>

As you build more complex agents, you'll need additional Python packages for features like database connections, API integrations, or specialized tools.

:::note[Environment Management]
Timbal automatically sets up a Python environment using `uv`. You can manage your project dependencies using `uv` commands just like in any other Python project. For example:

<CodeBlock language="bash" code ={`uv add package-name`}/>

This makes it easy to add new packages and manage your project's dependencies.
:::

## Part 1: Build a Simple Chatbot

🛠 Let's create a simple chatbot that can respond to user messages.

**1. Import the class `Agent` from the `timbal` package.**

<CodeBlock language="python" title="agent.py" code ={`from timbal import Agent`}/>

**2. Initialize an `Agent` object.**

<CodeBlock language="python" title="agent.py" code ={`agent = Agent(
    name="my_agent",
    model="gemini/gemini-2.5-pro-preview-03-25" # provider/model
)`}/>

The `name` parameter is required and provides a unique identifier for your agent. The `model` parameter specifies the provider and model to use for the agent.

**3. Set your environment variables**
Before running your flow, make sure you have the keys needed set as environment variables in your `.env` file:

👀 It will depend on the LLM you're using, in this case we are using a Gemini model
<CodeBlock language="bash" title=".env" code ={`GEMINI_API_KEY=your_api_key_here`}/>

Only with the `Agent` class we have a flow that represents a llm that receives a `prompt` and returns a `response`.

Now let's run the chatbot!

<CodeBlock language="python" title="agent.py"
code ={`response = await agent(prompt="What is the capital of Germany?").collect()
print(response.output.content[0].text)`}/>


You will see an output like this:

<CodeBlock language="bash" code ={`The capital of Germany is Berlin.`}/>


:::tip[Congratulations!]
You've just created your first Timbal flow!
:::

This is the **simplest flow** you can create.

You can modify it as you want. For example, you can add tools to the agent.

## Part 2: Enhancing the Chatbot with Tools

A great feature of Timbal is that you can easily add tools to your agent, allowing it to perform actions or fetch information in real time. 

You can use both <span style={{color: 'var(--timbal-purple)'}}><strong>prebuilt tools</strong></span> (already provided by Timbal or the community) and <span style={{color: 'var(--timbal-purple)'}}><strong>custom tools</strong></span> (functions you create yourself).

Let's see both approaches with practical examples:

### Example 1: Using a Prebuilt Tool

Timbal comes with several tools ready to use. Here are two powerful examples:

#### a) Internet Search

You can use the search tool to let your agent search the internet for up-to-date information:

<CodeBlock language="python" title="agent.py"
code ={`from timbal.steps.perplexity import search

agent = Agent(
    name="search_agent",
    model="gemini/gemini-2.5-pro-preview-03-25",
    tools=[search]
)`}/>

#### b) SQL/Postgres Query

Timbal also provides a prebuilt tool to query a Postgres database:

<CodeBlock language="python" title="agent.py"
code ={`from timbal.steps.sql.postgres import postgres_query, PGConfig

db_config = PGConfig(
    host="localhost",
    port=5432,
    user="youruser",
    password="yourpassword",
    database="yourdb"
)

def run_sql_query(sql: str) -> str:
    result = postgres_query(sql, db_config=db_config)
    return str(result)

agent = Agent(
    name="sql_agent",
    model="gemini/gemini-2.5-pro-preview-03-25",
    tools=[run_sql_query]
)`}/>


### Example 2: Creating Your Own Tool

You can also create your own tools. Here's a very simple example: a tool that returns the current date and time.

<CodeBlock language="python" title="agent.py"
code ={`from datetime import datetime

def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

agent = Agent(
    name="datetime_agent",
    model="gemini/gemini-2.5-pro-preview-03-25",
    tools=[get_datetime]
)`}/>

With Timbal, you can combine as many tools as you want—prebuilt or custom—to make your agent as powerful as you need!

**That's it!** With 5 lines of code we've created a chatbot with tools.

___

_Looking for the full list of integrations or examples?_

_Head over to [integrations](/integrations)/[examples](/examples) or, even better, check out our_ <a href="https://github.com/timbal-ai/timbal" target="_blank" rel="noopener noreferrer">GitHub</a>! _We're adding new examples as fast as we can._ 

_Want the latest and greatest? Just do a <b>git pull</b> to make sure you have all the newest content! 😄_

___

## Deploying Timbal

The best way to deploy your Timbal project is through the official Timbal Platform. It’s fast, easy, and gives you access to all the latest features and integrations.

👉 Go to the [Timbal Platform] to get started!

Prefer to run things locally or on your own infrastructure?

No problem! You can use `timbal build` to generate a ready-to-use Docker container with your flow and an HTTP server preconfigured. This makes it simple to deploy anywhere Docker runs.