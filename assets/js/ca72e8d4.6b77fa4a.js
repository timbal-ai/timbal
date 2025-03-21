"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[8935],{6536:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>d,default:()=>a,frontMatter:()=>r,metadata:()=>s,toc:()=>o});const s=JSON.parse('{"id":"concepts/step","title":"Steps","description":"Steps are the fundamental processing units within a workflow. They represent individual operations, from simple functions to complex language models, that can be connected together to create sophisticated workflows.","source":"@site/docs/concepts/step.md","sourceDirName":"concepts","slug":"/concepts/step","permalink":"/docs/concepts/step","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Steps","sidebar":"docsSidebar"}}');var l=n(4848),i=n(8453);const r={title:"Steps",sidebar:"docsSidebar"},d="Steps",c={},o=[{value:"Overview",id:"overview",level:2},{value:"Attributes",id:"attributes",level:2}];function h(e){const t={code:"code",h1:"h1",h2:"h2",header:"header",li:"li",p:"p",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",ul:"ul",...(0,i.R)(),...e.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(t.header,{children:(0,l.jsx)(t.h1,{id:"steps",children:"Steps"})}),"\n",(0,l.jsx)(t.p,{children:"Steps are the fundamental processing units within a workflow. They represent individual operations, from simple functions to complex language models, that can be connected together to create sophisticated workflows."}),"\n",(0,l.jsx)(t.h2,{id:"overview",children:"Overview"}),"\n",(0,l.jsx)(t.p,{children:"The Step class encapsulates a handler function and provides functionality to:"}),"\n",(0,l.jsxs)(t.ul,{children:["\n",(0,l.jsx)(t.li,{children:"Validate input parameters using Pydantic models"}),"\n",(0,l.jsx)(t.li,{children:"Validate return values using Pydantic models"}),"\n",(0,l.jsx)(t.li,{children:"Execute the handler function with proper parameter passing"}),"\n",(0,l.jsx)(t.li,{children:"Support both synchronous and asynchronous execution"}),"\n"]}),"\n",(0,l.jsx)(t.h2,{id:"attributes",children:"Attributes"}),"\n",(0,l.jsxs)(t.table,{children:[(0,l.jsx)(t.thead,{children:(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.th,{style:{textAlign:"left"},children:"Attribute"}),(0,l.jsx)(t.th,{style:{textAlign:"left"},children:"Parameter"}),(0,l.jsx)(t.th,{style:{textAlign:"left"},children:"Type"}),(0,l.jsx)(t.th,{style:{textAlign:"left"},children:"Description"})]})}),(0,l.jsxs)(t.tbody,{children:[(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.strong,{children:"Function"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"handler_fn"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"str"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:"The function that implements the step's processing logic."})]}),(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.strong,{children:"Input Parameters Model"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"handler_params_model"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"str"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:"The Pydantic model for validating input parameters."})]}),(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.strong,{children:"Output Parameters Model"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"handler_return_model"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"str"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:"The Pydantic model for validating return values."})]}),(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.strong,{children:"LLM"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"is_llm"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"bool"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:"Whether the step is a LLM."})]}),(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.strong,{children:"Async"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"is_coroutine"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"bool"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:"Whether the step is a coroutine."})]}),(0,l.jsxs)(t.tr,{children:[(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.strong,{children:"Async Generator"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"is_async_gen"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:(0,l.jsx)(t.code,{children:"bool"})}),(0,l.jsx)(t.td,{style:{textAlign:"left"},children:"Whether the step is an async generator."})]})]})]})]})}function a(e={}){const{wrapper:t}={...(0,i.R)(),...e.components};return t?(0,l.jsx)(t,{...e,children:(0,l.jsx)(h,{...e})}):h(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>r,x:()=>d});var s=n(6540);const l={},i=s.createContext(l);function r(e){const t=s.useContext(i);return s.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function d(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(l):e.components||l:r(e.components),s.createElement(i.Provider,{value:t},e.children)}}}]);