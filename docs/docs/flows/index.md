---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Understanding Flows

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.2rem', fontWeight: 'normal'}}>
Learn how Flows connect and coordinate tasks to create intelligent, end-to-end AI solutions.
</h2>

## What is a Flow?

Think of a Flow as a super-smart assembly line for your AI tasks! It's like having a team of specialized workers (steps) who pass information to each other in a carefully planned sequence.

<CodeBlock language="python" code ={`flow = Flow()`}/>

Now you have initialized a Flow! But this does not do nothing itself... We have to add functionalities! 

## The Building Blocks
A Flow is made up of two main components:

<div className="cards-container">
<div className="card">
<div className="card-content">

### Steps

These are your individual workers:
- Each step has a specific job to do
- They can be simple functions or complex AI models
- They process data and pass it along

</div>
</div>

<div className="card">
<div className="card-content">

### Links

These are the connections between steps:
- They tell your flow how to move data around
- They can even make decisions about which path to take!
- They define the flow of information

</div>
</div>
</div>

## What Can Flows Do?

<div className="capabilities">
<div className="capability">
<div className="capability-icon">
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g fill="transparent" stroke="var(--timbal-purple)" stroke-linecap="round" stroke-width="1.5"><path d="M10.046 14c-1.506-1.512-1.37-4.1.303-5.779l4.848-4.866c1.673-1.68 4.25-1.816 5.757-.305s1.37 4.1-.303 5.78l-2.424 2.433"/><path d="M13.954 10c1.506 1.512 1.37 4.1-.303 5.779l-2.424 2.433l-2.424 2.433c-1.673 1.68-4.25 1.816-5.757.305s-1.37-4.1.303-5.78l2.424-2.433"/></g></svg>
</div>
<div className="capability-content">
<h3>Connect Multiple Steps</h3>
<p>Create complex workflows, pass data between different components, and make everything work together smoothly.</p>
</div>
</div>

<div className="capability">
<div className="capability-icon">
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g fill="none" stroke="var(--timbal-purple)" stroke-width="1.5"><path stroke-linecap="round" d="M18 10h-5"/><path d="M10 3h6.5c.464 0 .697 0 .892.026a3 3 0 0 1 2.582 2.582c.026.195.026.428.026.892"/><path d="M2 6.95c0-.883 0-1.324.07-1.692A4 4 0 0 1 5.257 2.07C5.626 2 6.068 2 6.95 2c.386 0 .58 0 .766.017a4 4 0 0 1 2.18.904c.144.119.28.255.554.529L11 4c.816.816 1.224 1.224 1.712 1.495a4 4 0 0 0 .848.352C14.098 6 14.675 6 15.828 6h.374c2.632 0 3.949 0 4.804.77q.119.105.224.224c.77.855.77 2.172.77 4.804V14c0 3.771 0 5.657-1.172 6.828S17.771 22 14 22h-4c-3.771 0-5.657 0-6.828-1.172S2 17.771 2 14z"/></g></svg>
</div>
<div className="capability-content">
<h3>Handle Different Types of Work</h3>
<p>Run tasks synchronously or asynchronously, work with AI models, and create reusable components.</p>
</div>
</div>

<div className="capability">
<div className="capability-icon">
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g fill="none" stroke="var(--timbal-purple)" stroke-width="1.5"><path d="M7 10c0-1.414 0-2.121.44-2.56C7.878 7 8.585 7 10 7h4c1.414 0 2.121 0 2.56.44c.44.439.44 1.146.44 2.56v4c0 1.414 0 2.121-.44 2.56c-.439.44-1.146.44-2.56.44h-4c-1.414 0-2.121 0-2.56-.44C7 16.122 7 15.415 7 14z"/><path d="M4 12c0-3.771 0-5.657 1.172-6.828S8.229 4 12 4s5.657 0 6.828 1.172S20 8.229 20 12s0 5.657-1.172 6.828S15.771 20 12 20s-5.657 0-6.828-1.172S4 15.771 4 12Z"/><path stroke-linecap="round" d="M4 12H2m20 0h-2M4 9H2m20 0h-2M4 15H2m20 0h-2m-8 5v2m0-20v2M9 20v2M9 2v2m6 16v2m0-20v2"/></g></svg></div>
<div className="capability-content">
<h3>Manage Data</h3>
<p>Stream results in real-time, save progress between runs, and keep everything organized.</p>
</div>
</div>
</div>

## 🛠️ Let's Build a Simple Flow!

Ready to create your first flow? Let's do it step by step:

<CodeBlock language="python" code ={`# 1. Create a new flow
flow = Flow()

# 2. Add a runnable
flow.add_step(parse_documentation)

# 3. Add more steps as needed
flow.add_step(create_database)

# 4. Connect them with links
flow.add_link("create_database,text", "parse_documentation.return")

# 5. Return the output as we want
flow.set_output("status", "create_database.return")`}/>

## Running a Flow

Similar to Agent in order to run a Flow there are 2 ways depending on the synchronisation.

### Synchronous Output Mode

For when the agent returns a complete response after processing. We will use the `complete()` function:

<CodeBlock language="python" code ={`response = await flow.complete(location="Barcelona")`}/>

### Streaming Response

Otherwise, when we want to know specific information on each event we can find the response asynchrounsly by running `run()`:

<CodeBlock language="python" code ={`response = async for event in flow.run(locatin="Barcelona"):
    print(event) `}/>

Events tell you what's happening in your flow. Here's what you can do with them:

<CodeBlock language="python" code ={`async for event in flow.run():
    if event.type == "START":
        print(f"Starting Agent: {event.step_id}")`}/>

<CodeBlock language="python" code ={`async for event in flow.run():
    if event.type == "OUTPUT":
        print(f"Agent finished in {event.elapsed_time}ms")
        print(f"Outputs: {event.outputs}")`}/>

## Using Memory in Flows

When using a state saver, memory persists between flow executions.

To do it we have to use the function **compile()** and define and **state_saver** (You can find more information [here](/state))

<CodeBlock language="python" code ={`flow = (
      Flow()
      .add_llm(memory_id="persistent_memory")
      .compile(state_saver=InMemorySaver())
)`}/>

## Next Steps

- Try creating your own Flow
- Experiment with different configurations
- See an example agent in [Examples](/examples)


<style>{`
.cards-container {
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
  flex-wrap: wrap;
}

.card {
  flex: 1;
  min-width: 300px;
  background: var(--ifm-background-color);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  overflow: hidden;
}

.card-content {
  padding: 1.5rem;
}

.card-content h3 {
  color: var(--ifm-color-primary);
  margin-top: 0;
  margin-bottom: 1rem;
}

[data-theme='dark'] .card-content h3 {
  color: #9d7cff;
}

.card-content ul {
  list-style: disc;
  padding-left: 1.2em;
  margin: 0;
}

.card-content li {
  margin: 0.5rem 0;
}

.capabilities {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin: 2rem 0;
}

.capability {
  display: flex;
  align-items: flex-start;
  gap: 1.5rem;
  padding: 1.5rem;
  background: var(--ifm-background-color);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.capability-icon {
  font-size: 2rem;
  line-height: 1;
  flex-shrink: 0;
}

.capability-content {
  flex: 1;
}

.capability-content h3 {
  color: var(--ifm-font-color-base);
  margin-top: 0;
  margin-bottom: 0.5rem;
}

.capability-content p {
  margin: 0;
  line-height: 1.5;
}
`}</style>
