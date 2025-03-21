---
sidebar_position: 3
sidebar: 'docsSidebar'
---

# Installation

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.2rem', fontWeight: 'normal'}}>
Get started with Timbal - Install, configure, and build your first AI application.
</h2>

<br />

:::info[Python Version Requirements]
**Timbal** requires `Python >= 3.12`.

Make sure you have the correct version installed before proceeding.
:::

⚙️ Timbal uses the `uv` for dependency management and package handling, streamlining project setup and execution for a seamless development experience.

:::note[Recommendation]
Install uv to manage your Python dependencies.

- On macOS/Linux:
Use `curl` to download the script and execute it with `sh`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If your system doesn’t have `curl`, you can use `wget`:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
- On Windows:
Use `irm` to download the script and `iex` to execute it:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
If you run into any issues, refer to **[UV’s installation](https://docs.astral.sh/uv/getting-started/installation/)** guide for more information.
:::


### 🚀 Install Timbal

Run the following command to install Timbal CLI:

```bash
uv add timbal
```

:::success[Success]
You’re ready to create your first Timbal project! 🎉
:::

# Creating a Timbal Project

Here's how to get started:

**1. Initialize Project Structure**

- Run the `timbal` CLI command:

```bash
timbal init my-project
```

- This creates a new project with the following structure:

```bash
my-project/
├── flow.py # Your main AI flow logic
├── .dockerignore # Ignore files when building Docker images
├── pyproject.toml # Project dependencies and settings
└── timbal.yaml # Configuration file for Timbal
```

**2. Personalize Your Project**

You will have a file `flow.py` with a basic structure of a flow.
This is the only file you need to edit to create your own flow.

```python
from datetime import datetime

from timbal import Agent
from timbal.state.savers import TimbalPlatformSaver

def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

flow = Agent(
    model="gpt-4o-mini",
    tools=[get_datetime],
).compile(state_saver=TimbalPlatformSaver())
```

#### ??????? posar algo del .env pero idk
:::note[Recommendation]
Store sensitive information like API keys in a `.env` file.
:::

**3. Run Your Flow**

Before you run your flow, make sure to run:

```bash
timbal install
```

Need additional packages? Add them using `uv`:  

```shell
uv add <package-name>
```

To run your flow, execute the following command in the root of your project:

```bash
timbal run
```
