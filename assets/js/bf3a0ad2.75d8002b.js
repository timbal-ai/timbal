"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9552],{7213:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>l,contentTitle:()=>r,default:()=>h,frontMatter:()=>a,metadata:()=>s,toc:()=>c});const s=JSON.parse('{"id":"get-started/quickstart","title":"Quickstart","description":"Build your first flow with Timbal with 5 lines of code.","source":"@site/docs/get-started/quickstart.md","sourceDirName":"get-started","slug":"/get-started/quickstart","permalink":"/timbal/docs/get-started/quickstart","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"sidebar_position":2,"sidebar":"docsSidebar"},"sidebar":"docsSidebar","previous":{"title":"Installation","permalink":"/timbal/docs/get-started/installation"}}');var o=n(4848),i=n(8453);const a={sidebar_position:2,sidebar:"docsSidebar"},r="Quickstart",l={},c=[{value:"Part 1: Build a Simple Chatbot",id:"part-1-build-a-simple-chatbot",level:2},{value:"Part 2: Enhancing the Chatbot with Tools",id:"part-2-enhancing-the-chatbot-with-tools",level:2},{value:"Part 3: Enhancing the Chatbot with Memory",id:"part-3-enhancing-the-chatbot-with-memory",level:2}];function d(e){const t={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",header:"header",p:"p",pre:"pre",strong:"strong",...(0,i.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(t.header,{children:(0,o.jsx)(t.h1,{id:"quickstart",children:"Quickstart"})}),"\n",(0,o.jsx)("h2",{className:"subtitle",style:{marginTop:"-17px",fontSize:"1.2rem",fontWeight:"normal"},children:(0,o.jsx)(t.p,{children:"Build your first flow with Timbal with 5 lines of code."})}),"\n",(0,o.jsx)("br",{}),"\n",(0,o.jsxs)(t.p,{children:["We'll start implementing an ",(0,o.jsx)(t.strong,{children:"agent"}),".\ud83e\udd16 It will be a ",(0,o.jsx)(t.strong,{children:"simple chatbot"})," and gradually enhance it with advanced features. Let's dive in! \ud83c\udf1f"]}),"\n",(0,o.jsxs)(t.p,{children:["Before moving forward, ensure you've completed the installation of Timbal. If you haven't set it up yet, follow the ",(0,o.jsx)(t.strong,{children:(0,o.jsx)(t.a,{href:"./installation",children:"installation guide"})})," to get started."]}),"\n",(0,o.jsx)(t.h2,{id:"part-1-build-a-simple-chatbot",children:"Part 1: Build a Simple Chatbot"}),"\n",(0,o.jsx)(t.p,{children:"\ud83d\udee0 Let's create a simple chatbot that can respond to user messages."}),"\n",(0,o.jsx)(t.p,{children:(0,o.jsxs)(t.strong,{children:["1. Import the class ",(0,o.jsx)(t.code,{children:"Flow"})," from the ",(0,o.jsx)(t.code,{children:"timbal"})," package."]})}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:"from timbal import Agent\n"})}),"\n",(0,o.jsx)(t.p,{children:(0,o.jsxs)(t.strong,{children:["1. Initialize a ",(0,o.jsx)(t.code,{children:"Flow"})," object."]})}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:"flow = Agent()\n"})}),"\n",(0,o.jsxs)(t.p,{children:[(0,o.jsx)(t.strong,{children:"2. Set your environment variables"}),"\nBefore running your flow, make sure you have the keys needed set as environment variables in your ",(0,o.jsx)(t.code,{children:".env"})," file:"]}),"\n",(0,o.jsx)(t.p,{children:"\ud83d\udc40 It will depend on the LLM you're using, in this case, the default model is OpenAI."}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{children:"OPENAI_API_KEY=...\n"})}),"\n",(0,o.jsxs)(t.p,{children:["Only with the ",(0,o.jsx)(t.code,{children:"Agent"})," class we have a flow that represents a llm that receives a ",(0,o.jsx)(t.code,{children:"prompt"})," and returns a ",(0,o.jsx)(t.code,{children:"response"}),"."]}),"\n",(0,o.jsx)(t.p,{children:"Now let's run the chatbot!"}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:'response = flow.complete(prompt="What is the capital of Germany?")\nprint(response.content[0].text)\n'})}),"\n",(0,o.jsx)(t.p,{children:"You will see an output like this:"}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{children:"The capital of Germany is Berlin.\n"})}),"\n",(0,o.jsx)(t.admonition,{title:"Congratulations!",type:"tip",children:(0,o.jsx)(t.p,{children:"You've just created your first Timbal flow!"})}),"\n",(0,o.jsx)(t.p,{children:"This is the simplest flow you can create."}),"\n",(0,o.jsx)(t.p,{children:"You can modify id as you want. For example, you can add tools to the agent."}),"\n",(0,o.jsx)(t.h2,{id:"part-2-enhancing-the-chatbot-with-tools",children:"Part 2: Enhancing the Chatbot with Tools"}),"\n",(0,o.jsx)(t.p,{children:"When the chatbot encounters questions it can\u2019t answer from memory, we\u2019ll equip it with a tool. This allows the bot to fetch relevant information in real time, improving its responses. \ud83d\ude80"}),"\n",(0,o.jsx)(t.p,{children:"For this example, we will use a tool capable of returning the current weather."}),"\n",(0,o.jsx)(t.p,{children:(0,o.jsxs)(t.strong,{children:["1. Define the ",(0,o.jsx)(t.code,{children:"get_weather"})," tool."]})}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:"def get_weather() -> str:\n    ...\n"})}),"\n",(0,o.jsxs)(t.p,{children:[(0,o.jsx)(t.strong,{children:"2. Add an agent node to the flow with the tool."}),"\nWe need to add the tool to the llm as a new node."]}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:'flow.add_agent(model="gpt-4o-mini", memory_id="agent", tools=[get_weather])\n'})}),"\n",(0,o.jsxs)(t.p,{children:[(0,o.jsx)(t.strong,{children:"3. Set the data map of the agent to the prompt."}),"\nWe need to pass the prompt to the agent."]}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:'flow.set_data_map("agent.prompt", "prompt")\n'})}),"\n",(0,o.jsxs)(t.p,{children:[(0,o.jsx)(t.strong,{children:"4. Return the response as we want"}),"\nWe can map the response to the output of the flow."]}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:'flow.set_output("response", "agent.return")\n'})}),"\n",(0,o.jsx)(t.p,{children:"Here's the full code:"}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'{6} title="flow.py"',children:'from timbal import Flow\nfrom timbal.state.savers import InMemorySaver\n\nflow = (\n    Flow()\n    .add_agent(model="gpt-4o-mini", memory_id="agent", tools=[get_weather])\n    .set_data_map("agent.prompt", "prompt")\n    .set_output("response", "agent.return")\n)\n'})}),"\n",(0,o.jsx)(t.p,{children:"Let's visualize the graph we've built."}),"\n",(0,o.jsx)(t.admonition,{title:"Visualize the flow",type:"tip",children:(0,o.jsxs)(t.p,{children:["You can visualize the flow by calling ",(0,o.jsx)(t.code,{children:"flow.plot()"}),"."]})}),"\n",(0,o.jsx)("div",{align:"center",children:(0,o.jsx)("img",{src:"../../static/img/dag_tools.png",style:{width:"50%"}})}),"\n",(0,o.jsx)(t.p,{children:"But it does not retain the conversation:"}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{children:"user: My name is David\nassistant: Hello David, how can I help you today?\nuser: What is my name?\nassistant: I don't know but you can tell me.\n"})}),"\n",(0,o.jsx)(t.p,{children:"Let's add memory to the chatbot."}),"\n",(0,o.jsx)(t.h2,{id:"part-3-enhancing-the-chatbot-with-memory",children:"Part 3: Enhancing the Chatbot with Memory"}),"\n",(0,o.jsx)(t.p,{children:"\ud83e\udde0 Let's add memory to the chatbot."}),"\n",(0,o.jsx)(t.p,{children:"It is very simple, we just need to set a context."}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:"from timbal.state.context import RunContext\n"})}),"\n",(0,o.jsxs)(t.p,{children:["Timbal offers you a ",(0,o.jsx)(t.code,{children:"RunContext"})," object that you can use to store the conversation history.\nYou can manage the parameters: ",(0,o.jsx)(t.code,{children:"model_config"}),", ",(0,o.jsx)(t.code,{children:"id"}),", ",(0,o.jsx)(t.code,{children:"parent_id"})," and ",(0,o.jsx)(t.code,{children:"timbal_platform_config"}),"."]}),"\n",(0,o.jsxs)(t.p,{children:["Only you have to pass to the flow the ",(0,o.jsx)(t.code,{children:"RunContext"})," object."]}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-python",metastring:'title="flow.py"',children:"run_context = RunContext()\nflow_output_event = await flow.complete(context=run_context, prompt=prompt)\n"})}),"\n",(0,o.jsxs)(t.p,{children:["As you can see the parameter ",(0,o.jsx)(t.code,{children:"context"})," is initialized with the ",(0,o.jsx)(t.code,{children:"RunContext"})," object."]}),"\n",(0,o.jsx)(t.p,{children:"And the previous conversation will look like:"}),"\n",(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{children:"user: My name is David\nassistant: Hello David, how can I help you today?\nuser: What is my name?\nassistant: Your name is David. Do you need anything else?\n"})}),"\n",(0,o.jsxs)(t.p,{children:[(0,o.jsx)(t.strong,{children:"That's it!"})," \ud83d\udca5 With 5 lines of code we've created a chatbot that indeed can answer questions about the current time."]})]})}function h(e={}){const{wrapper:t}={...(0,i.R)(),...e.components};return t?(0,o.jsx)(t,{...e,children:(0,o.jsx)(d,{...e})}):d(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>a,x:()=>r});var s=n(6540);const o={},i=s.createContext(o);function a(e){const t=s.useContext(i);return s.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function r(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:a(e.components),s.createElement(i.Provider,{value:t},e.children)}}}]);