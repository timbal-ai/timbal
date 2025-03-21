"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4333],{767:(e,t,l)=>{l.r(t),l.d(t,{assets:()=>c,contentTitle:()=>d,default:()=>x,frontMatter:()=>r,metadata:()=>n,toc:()=>o});const n=JSON.parse('{"id":"concepts/llm","title":"LLMs","description":"A comprehensive guide to configuring and using Large Language Models (LLMs) in your Timbal projects.","source":"@site/docs/concepts/llm.md","sourceDirName":"concepts","slug":"/concepts/llm","permalink":"/docs/concepts/llm","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"LLMs","sidebar":"docsSidebar"}}');var s=l(4848),i=l(8453);const r={title:"LLMs",sidebar:"docsSidebar"},d="LLMs",c={},o=[{value:"What are LLMs?",id:"what-are-llms",level:2},{value:"Setting Up",id:"setting-up",level:3},{value:"Attributes",id:"attributes",level:3}];function h(e){const t={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",ul:"ul",...(0,i.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(t.header,{children:(0,s.jsx)(t.h1,{id:"llms",children:"LLMs"})}),"\n",(0,s.jsx)(t.p,{children:"A comprehensive guide to configuring and using Large Language Models (LLMs) in your Timbal projects."}),"\n",(0,s.jsx)(t.h2,{id:"what-are-llms",children:"What are LLMs?"}),"\n",(0,s.jsx)(t.p,{children:"LLMs are the core of Timbal. They are the ones that will be used to create agents with tools. They enable agents to understand context, make decisions, and generate human-like responses. Here's what you need to know:"}),"\n",(0,s.jsxs)(t.p,{children:["As you have seen in ",(0,s.jsx)(t.a,{href:"/docs/concepts/flow/",children:"Flows"}),", LLMs behaves as steps that there function is to call an LLM provider."]}),"\n",(0,s.jsx)(t.p,{children:"So the way to use an LLM is to create a flow with an LLM step."}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{className:"language-python",children:"from timbal import Flow\n\nflow = (Flow()\n    .add_llm()\n)\n"})}),"\n",(0,s.jsx)(t.h3,{id:"setting-up",children:"Setting Up"}),"\n",(0,s.jsx)(t.p,{children:"For Timbal to use an LLM, you need to set up the API key of the provider do you want the model to be used from. (e.g. OpenAI, Anthropic, TogetherAI or Gemini)"}),"\n",(0,s.jsx)(t.admonition,{type:"warning",children:(0,s.jsx)(t.p,{children:"Never commit API keys to version control. Use environment files (.env) or your system's secret management."})}),"\n",(0,s.jsx)(t.h3,{id:"attributes",children:"Attributes"}),"\n",(0,s.jsx)(t.admonition,{type:"tip",children:(0,s.jsx)(t.p,{children:"You don't have to worry about the options of the LLM. Timbal will take care of it for you."})}),"\n",(0,s.jsxs)(t.p,{children:["By specifying the ",(0,s.jsx)(t.code,{children:"model"})," parameter, the kwargs parameters will be in function of the LLM provider."]}),"\n",(0,s.jsxs)(t.table,{children:[(0,s.jsx)(t.thead,{children:(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.th,{style:{textAlign:"left"},children:"Attribute"}),(0,s.jsx)(t.th,{style:{textAlign:"left"},children:"Parameter"}),(0,s.jsx)(t.th,{style:{textAlign:"left"},children:"Type"}),(0,s.jsx)(t.th,{style:{textAlign:"left"},children:"Description"}),(0,s.jsx)(t.th,{style:{textAlign:"left"},children:"Provider Support"})]})}),(0,s.jsxs)(t.tbody,{children:[(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Prompt"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"prompt"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"str"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"The first input to send to the LLM."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"System Prompt"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"system_prompt"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"str"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"System prompt to guide the LLM's behavior and role."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Model"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"model"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"str"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Name of the LLM model to use."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Tools"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"tools"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"list[Tool | dict]"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"List of tools/functions the LLM can call."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Tool Choice"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"tool_choice"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"dict[str, Any] | str"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"How the model should use the provided tools."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Max Tokens"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"max_tokens"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"int"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"The maximum number of tokens in the response."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Temperature"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"temperature"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"float"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Sampling temperature (0-2 except for Anthropic which is 0-1)."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Frequency Penalty"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"frequency_penalty"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"float"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Penalty for token frequency."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI, TogetherAI"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Presence Penalty"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"presence_penalty"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"float"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Penalty for token presence."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Top P"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"top_p"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"float"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Nucleus sampling parameter."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI, TogetherAI, Gemini"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Top K"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"top_k"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"int"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Only sample from the top K options for each token."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Anthropic"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Logprobs"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"logprobs"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"bool"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Whether to return logprobs with the returned text."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Top Logprobs"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"top_logprobs"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"int"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Return log probabilities of the top N tokens."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI, TogetherAI"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Seed"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"seed"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"int"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Deterministic sampling parameter."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI, TogetherAI, Gemini"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Stop"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"stop"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"str | list[str]"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Up to 4 sequences where the model will stop generating."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"Parallel Tool Calls"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"parallel_tool_calls"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"bool"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"Whether to execute tool calls in parallel."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"OpenAI, TogetherAI"})]}),(0,s.jsxs)(t.tr,{children:[(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.strong,{children:"JSON Schema"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"json_schema"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:(0,s.jsx)(t.code,{children:"dict"})}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"JSON schema for structured output."}),(0,s.jsx)(t.td,{style:{textAlign:"left"},children:"All providers"})]})]})]}),"\n",(0,s.jsxs)(t.admonition,{type:"info",children:[(0,s.jsx)(t.p,{children:"These are some of the models that could be used:"}),(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsx)(t.li,{children:"OpenAI: gpt-4o, gpt-4o-mini, o1, o3-mini, o1-mini"}),"\n",(0,s.jsx)(t.li,{children:"Anthropic: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307"}),"\n",(0,s.jsx)(t.li,{children:"TogetherAI: deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.3-70B-Instruct-Turbo, meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo, meta-llama/Llama-3.2-3B-Instruct-Turbo,Qwen/Qwen2.5-Coder-32B-Instruct, Qwen/Qwen2-VL-72B-Instruct, mistralai/Mistral-Small-24B-Instruct-2501, mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mixtral-8x22B-Instruct-v0.1, meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"}),"\n",(0,s.jsx)(t.li,{children:"Gemini: gemini-2.0-flash-lite-preview-02-05, gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-flash-8b, text-embedding-004"}),"\n"]})]})]})}function x(e={}){const{wrapper:t}={...(0,i.R)(),...e.components};return t?(0,s.jsx)(t,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},8453:(e,t,l)=>{l.d(t,{R:()=>r,x:()=>d});var n=l(6540);const s={},i=n.createContext(s);function r(e){const t=n.useContext(i);return n.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function d(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:r(e.components),n.createElement(i.Provider,{value:t},e.children)}}}]);