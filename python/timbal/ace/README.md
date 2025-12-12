# Action Control Engine (ACE)

## Paper

### Intro
The Non-Determinism Problem**
- LLMs are fundamentally stochastic (temperature, sampling, etc.)
- Same input can yield different outputs across runs
- This is a feature for creativity, but a bug for reliability

**From LLMs to Agents**
- When LLMs can take *actions* (tool calls, API requests, side effects) → agents
- The non-determinism now has real-world consequences
- Wrong tool call = wrong email sent, wrong database write, wrong API hit

**Current State of Agent Building**
- Agents are built with implicit trust in LLM decision-making
- Developers write tools, prompts, and hope for the best
- Testing is ad-hoc: run it a few times, check if it "looks right"

**Why Evals Exist (but aren't enough)**
- Evals verify *output correctness* after the fact
- They answer: "Did the agent do the right thing?"
- They don't answer: "Can we *ensure* it does the right thing?"
- Evals are passive observers, not active controllers

**The Core Question**
- Can we do better than probabilistic hope?
- Can we introduce **controlled determinism** into agent execution?
- → ACE: Action Control Engine

### Related work
- Agent frameworks (LangChain, AutoGPT, etc.) - execution without control
- Eval frameworks (LMYS, Inspect, etc.) - observation without intervention
- Guardrails/Constitutional AI - output filtering, not execution control
- (something else about latest approaches remotely similar to ours)
- ACE sits in a new space: **execution-time behavioral control**

### ACE 
#### Evals framework (DSL, validators, ...)
...
#### Matching (bunch of approaches here -> we'll compare them in the next section)
- Context representation / state representation
...
#### Execution flow
...

### Experiments & testing & results & whatever (farem benchmarks i tal)
#### Benchmarks with other approaches / papers / companies

### Conclusions
- Contributions
- Limitations
- Future work

(somewhere we should talk about using the feedback for enriching the eval set thus making your ACE better)


```python
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..evals.models import Eval


class BaseAce(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    evals: list[Eval]

    @abstractmethod
    def forward(self, **kwargs) -> Any:
        # Matching -> Eval | None
        # if None -> base_execution_flow
        # else -> Eval.to_execution_flow
        ...
```

The ACE only applies to agents. Workflows are deterministic per se.
The ACE runs after the pre-hook
We need a mechanism to modify the execution flow of the agent

## Pseudo-deterministic Matching (nom fancy per aixo)

### Baseline 

Input embedding + evals inputs embeddings

-> max embedding similarity (we can add BM25, reranking, and other stuff later)

## Modifying the Execution Flow

### Baseline

seq!: ... -> Map this into TodoWrite example (THIS MAPPING IS DETERMINISTIC)

Force a TodoWrite tool use + result cycle

-> Base execution flow (with the todo write and todo result in the context)

### Brainstorming

def base_execution_flow(agent):
    while True:
        llm <- agent._llm
        output = llm()

        if not output.tools:
            break

        for tool in output.tools:
            tool <- agent.tools[tool]
            tool()

---

seq!:
    - send_email

def custom_execution_flow(agent):
    tool <- agent.tools['send_email']
    tool()

---

parallel!:
    - send_email
    - send_sms

def custom_execution_flow(agent):
    tools <- agent.tools['send_email', 'send_sms']
    asyncio.gather(*[tool() for tool in tools])

---

exemple senzill customer success (vicio)

seq!:
    - llm:
        output: (or input idk)
            prompt!: should say something like this: ...

def custom_execution_flow(agent):
    llm <- agent._llm
    output = llm(prompt=user prompt + example)
    (most of the times the input prompts won't match exactly with the eval reference)
