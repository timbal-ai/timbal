---
sidebar_position: 2
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Quickstart

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.2rem', fontWeight: 'normal'}}>
Build your first flow with Timbal with 5 lines of code.
</h2>

We'll start implementing an **agent**. It will be a **simple chatbot** and gradually enhance it with advanced features. Let's dive in!

Before moving forward, ensure you've completed the installation of Timbal. If you haven't set it up yet, follow the **[installation guide](/getting-started/installation)** to get started.



## Part 0: Creating a Timbal Project

Timbal makes your life easier by automatically setting up a complete project structure. When you run `timbal init`, it creates a directory with the following structure:

<div class="vertical-stepper">
  <div class="step">
    <div class="circle">1</div>
    <div class="step-content">
      <div class="step-title">Initialize Project Structure</div>
      <div class="step-desc">
      - Run the `timbal` CLI command:

        <CodeBlock language="bash" code ={`timbal init my-project`}/>

- This creates a new project with the following structure:
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
            <img src="/img/folder.svg" className="file-tree-icon" style={{ width: '14px' }} /> my-project
            <div className="file-list" style={{ marginLeft: '1.5rem' }}>
              <div><img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px' }} /> agent.py</div>
              <div><img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px' }} /> .dockerignore</div>
              <div><img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px' }} /> pyproject.toml</div>
              <div><img src="/img/file.svg" className="file-tree-icon" style={{ width: '14px' }} /> timbal.yaml</div>
            </div>
          </div>
          <div style={{ minWidth: '320px', flex: 1 }}>
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><b>agent.py</b></td>
                  <td>Main entry point for your AI flow logic. Define your agents, tools, and workflow here.</td>
                </tr>
                <tr>
                  <td><b>.dockerignore</b></td>
                  <td>Lists files and directories to exclude when building Docker images.</td>
                </tr>
                <tr>
                  <td><b>pyproject.toml</b></td>
                  <td>Manages project dependencies and settings. Used by Python tools.</td>
                </tr>
                <tr>
                  <td><b>timbal.yaml</b></td>
                  <td>Main configuration file for Timbal, where you set up project options and integrations.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>


</div>
    </div>
  </div>
  <div class="step">
    <div class="circle">2</div>
    <div class="step-content">
      <div class="step-title">Personalize Your Project</div>
      <div class="step-desc">
      You will have a file `agent.py` with a basic structure of a flow.
This is the only file you need to edit to create your own flow.

<CodeBlock language="python" code ={`from datetime import datetime

from timbal import Agent
from timbal.state.savers import TimbalPlatformSaver

def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

flow = Agent(
    model="gemini-2.5-pro-preview-03-25",
    tools=[get_datetime],
    state_saver=TimbalPlatformSaver()
)`}/>


👀 If you want to use LLM models, you might have to provide your API keys for the LLM providers corresponding to the models you are using.

:::note[Recommendation]
Store sensitive information like API keys in a `.env` file.
:::
</div>
    </div>
  </div>
  <div class="step">
    <div class="circle">3</div>
    <div class="step-content">
      <div class="step-title"> Run Your Flow</div>
      <div class="step-desc">
      </div>
    </div>
  </div>
</div>
:::note[Environment Management]
Timbal automatically sets up a Python environment using `uv`. You can manage your project dependencies using `uv` commands just like in any other Python project. For example:

<CodeBlock language="bash" code={`uv add package-name`}/>

This makes it easy to add new packages and manage your project's dependencies.
:::

## Part 1: Build a Simple Chatbot

🛠 Let's create a simple chatbot that can respond to user messages.

**1. Import the class `Agent` from the `timbal` package.**

<CodeBlock language="python" title="flow.py" code ={`from timbal import Agent`}/>

**2. Initialize an `Agent` object.**

<CodeBlock language="python" title="flow.py" code={`flow = Agent(
    model="gemini-2.5-pro-preview-03-25"
)`}/>

**3. Set your environment variables**
Before running your flow, make sure you have the keys needed set as environment variables in your `.env` file:

👀 It will depend on the LLM you're using, in this case we are using a Gemini model
<CodeBlock language="bash" title=".env" code={`GEMINI_API_KEYY=your_api_key_here `}/>

Only with the `Agent` class we have a flow that represents a llm that receives a `prompt` and returns a `response`.

Now let's run the chatbot!

<CodeBlock language="python" title="flow.py"
code={`response = flow.complete(prompt="What is the capital of Germany?")
print(response.content[0].text)`}/>


You will see an output like this:


The capital of Germany is Berlin.


:::tip[Congratulations!]
You've just created your first Timbal flow!
:::

This is the simplest flow you can create.

You can modify it as you want. For example, you can add tools to the agent.

## Part 2: Enhancing the Chatbot with Tools

A great feature of Timbal is that you can easily add tools to your agent, allowing it to perform actions or fetch information in real time. You can use both prebuilt tools (already provided by Timbal or the community) and custom tools (functions you create yourself).

Let's see both approaches with practical examples:

### Example 1: Using a Prebuilt Tool

Timbal comes with several tools ready to use. Here are two powerful examples:

#### a) Internet Search

You can use the search tool to let your agent search the internet for up-to-date information:

<CodeBlock language="python" title="agent.py"
code={`from timbal.steps.perplexity import search

flow = Agent(
    tools=[search]
)`}/>

#### b) SQL/Postgres Query

Timbal also provides a prebuilt tool to query a Postgres database:

<CodeBlock language="python" title="agent.py"
code={`from timbal.steps.sql.postgres import postgres_query, PGConfig

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

flow = Agent(
    tools=[run_sql_query]
)`}/>


### Example 2: Creating Your Own Tool

You can also create your own tools. Here's a very simple example: a tool that returns the current date and time.

<CodeBlock language="python" title="agent.py"
code={`from datetime import datetime

def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

flow = Agent(
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

👉 Go to the [Timbal Platform](/platform) to get started!

Prefer to run things locally or on your own infrastructure?

No problem! You can use `timbal build` to generate a ready-to-use Docker container with your flow and an HTTP server preconfigured. This makes it simple to deploy anywhere Docker runs.