"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[209],{384:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>l,contentTitle:()=>o,default:()=>h,frontMatter:()=>a,metadata:()=>s,toc:()=>c});const s=JSON.parse('{"id":"guides/streaming","title":"Streaming","description":"Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.","source":"@site/docs/guides/streaming.md","sourceDirName":"guides","slug":"/guides/streaming","permalink":"/timbal/docs/guides/streaming","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Streaming","sidebar":"docsSidebar"}}');var i=t(4848),r=t(8453);const a={title:"Streaming",sidebar:"docsSidebar"},o="Streaming",l={},c=[{value:"Working with Events",id:"working-with-events",level:3}];function d(e){const n={admonition:"admonition",code:"code",h1:"h1",h3:"h3",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,r.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.header,{children:(0,i.jsx)(n.h1,{id:"streaming",children:"Streaming"})}),"\n",(0,i.jsx)(n.p,{children:"Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs."}),"\n",(0,i.jsx)(n.p,{children:"There are two ways I could stream the output of a flow:"}),"\n",(0,i.jsx)(n.p,{children:"If I want to see the output as it's being generated."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"Hi my name is... \n"})}),"\n",(0,i.jsxs)(n.ol,{children:["\n",(0,i.jsxs)(n.li,{children:["Using the ",(0,i.jsx)(n.code,{children:"run()"})," method, which returns an async iterator that yields events as the flow executes."]}),"\n"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"async for event in flow.run():\n    print(event)\n"})}),"\n",(0,i.jsx)(n.p,{children:"Otherwise I want to see the final result of the flow."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"Hi my name is David. How are you?\n"})}),"\n",(0,i.jsxs)(n.ol,{start:"2",children:["\n",(0,i.jsxs)(n.li,{children:["Using the ",(0,i.jsx)(n.code,{children:"complete()"})," method, which returns the final result of the flow."]}),"\n"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"result = await flow.complete()\n"})}),"\n",(0,i.jsxs)(n.admonition,{title:"Important \u26a0\ufe0f",type:"warning",children:[(0,i.jsx)(n.p,{children:"Remember: These are coroutines! You need to:"}),(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["Use ",(0,i.jsx)(n.code,{children:"await"})]}),"\n",(0,i.jsx)(n.li,{children:"Run them in an async function"}),"\n"]})]}),"\n",(0,i.jsx)(n.h3,{id:"working-with-events",children:"Working with Events"}),"\n",(0,i.jsx)(n.p,{children:"Events tell you what's happening in your flow. Here's what you can do with them:"}),"\n",(0,i.jsx)(n.p,{children:"If you want to know when a step starts..."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'async for event in flow.run():\n    if event.type == "STEP_START":\n        print(f"Starting step: {event.step_id}")\n'})}),"\n",(0,i.jsx)(n.p,{children:"If you want to see output as it's generated..."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'async for event in flow.run():\n    if event.type == "STEP_CHUNK":\n        print(event.step_chunk, end="")  # Print each piece as it arrives\n'})}),"\n",(0,i.jsx)(n.p,{children:"If you want to know step results and timing..."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'async for event in flow.run():\n    if event.type == "STEP_OUTPUT":\n        print(f"Step completed in {event.elapsed_time}ms")\n        print(f"Result: {event.step_result}")\n'})}),"\n",(0,i.jsx)(n.p,{children:"If you want the final flow results..."}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'async for event in flow.run():\n    if event.type == "FLOW_OUTPUT":\n        print(f"Flow finished in {event.elapsed_time}ms")\n        print(f"Outputs: {event.outputs}")\n'})})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>a,x:()=>o});var s=t(6540);const i={},r=s.createContext(i);function a(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:a(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);