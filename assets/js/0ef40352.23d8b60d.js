"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5834],{2824:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>l,contentTitle:()=>o,default:()=>h,frontMatter:()=>a,metadata:()=>t,toc:()=>d});const t=JSON.parse('{"id":"guides/saver","title":"Saver","description":"State savers in Timbal provide persistence mechanisms for storing and retrieving flow execution states. They enable memory retention across sessions and allow you to implement custom storage solutions.","source":"@site/docs/guides/saver.md","sourceDirName":"guides","slug":"/guides/saver","permalink":"/timbal/docs/guides/saver","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Saver","sidebar":"docsSidebar"}}');var r=s(4848),i=s(8453);const a={title:"Saver",sidebar:"docsSidebar"},o="State Savers",l={},d=[{value:"Built-in State Savers",id:"built-in-state-savers",level:2},{value:"InMemorySaver",id:"inmemorysaver",level:3},{value:"JSONLSaver",id:"jsonlsaver",level:3},{value:"Creating Custom State Savers",id:"creating-custom-state-savers",level:2},{value:"The Snapshot Model",id:"the-snapshot-model",level:3}];function c(e){const n={code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,i.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"state-savers",children:"State Savers"})}),"\n",(0,r.jsx)(n.p,{children:"State savers in Timbal provide persistence mechanisms for storing and retrieving flow execution states. They enable memory retention across sessions and allow you to implement custom storage solutions."}),"\n",(0,r.jsx)(n.h2,{id:"built-in-state-savers",children:"Built-in State Savers"}),"\n",(0,r.jsx)(n.p,{children:"Timbal includes some built-in state savers:"}),"\n",(0,r.jsx)(n.h3,{id:"inmemorysaver",children:"InMemorySaver"}),"\n",(0,r.jsx)(n.p,{children:"The simplest state saver that keeps everything in memory. Useful for development and testing:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'from timbal import Flow\nfrom timbal.state.savers import InMemorySaver\n\nflow = (\n   Flow()\n   .add_llm(memory_id="conversation")\n   .compile(state_saver=InMemorySaver())\n   )\n'})}),"\n",(0,r.jsx)(n.h3,{id:"jsonlsaver",children:"JSONLSaver"}),"\n",(0,r.jsx)(n.p,{children:"Persists states to a JSONL file, providing simple file-based storage:"}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:'from timbal.state.savers import JSONLSaver\n\nflow = (\n   Flow()\n   .add_llm(memory_id="conversation")\n   .compile(state_saver=JSONLSaver("path/to/states.jsonl"))\n   )\n'})}),"\n",(0,r.jsx)(n.h2,{id:"creating-custom-state-savers",children:"Creating Custom State Savers"}),"\n",(0,r.jsxs)(n.p,{children:["You can create your own state saver by inheriting from ",(0,r.jsx)(n.code,{children:"BaseSaver"}),". You just need to implement three key methods:"]}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.code,{children:"get(id: str) -> Snapshot | None"})}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:"Retrieves a specific snapshot by ID"}),"\n",(0,r.jsx)(n.li,{children:"Returns None if not found"}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.code,{children:"get_last(n: int = 1, parent_id: str | None = None, group_id: str | None = None) -> list[Snapshot]"})}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:"Retrieves the last n snapshots matching criteria"}),"\n",(0,r.jsx)(n.li,{children:"Supports filtering by parent_id and group_id"}),"\n",(0,r.jsx)(n.li,{children:"Returns snapshots in chronological order"}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(n.li,{children:["\n",(0,r.jsx)(n.p,{children:(0,r.jsx)(n.code,{children:"put(snapshot: Snapshot) -> None"})}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:"Stores a new snapshot"}),"\n",(0,r.jsx)(n.li,{children:"Should assign UUID if snapshot.id is None"}),"\n",(0,r.jsx)(n.li,{children:"Must prevent duplicate IDs"}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(n.h3,{id:"the-snapshot-model",children:"The Snapshot Model"}),"\n",(0,r.jsxs)(n.p,{children:["The ",(0,r.jsx)(n.code,{children:"Snapshot"})," class represents a point-in-time state of a flow:"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"class Snapshot(BaseModel):\n   id: str | None = None # Unique identifier\n   parent_id: str | None = None # ID of parent snapshot\n   group_id: str | None = None # Group identifier\n   data: dict[str, Any] # State data\n   metadata: dict[str, Any] # Additional metadata\n"})})]})}function h(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(c,{...e})}):c(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>a,x:()=>o});var t=s(6540);const r={},i=t.createContext(r);function a(e){const n=t.useContext(i);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:a(e.components),t.createElement(i.Provider,{value:n},e.children)}}}]);