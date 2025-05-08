// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    {
      type: 'category',
      collapsed: false,
      label: 'Get Started',
      items: ['get-started/index', 'get-started/installation', 'get-started/quickstart', 'get-started/model_capabilities'],
    },
    {
      type: 'category',
      label: 'Concepts',
      collapsed: false,
      items: [
        'concepts/index',
        'concepts/agents',
        'concepts/flows',
        //'concepts/knowledge_bases',
        //'concepts/memories',
        'concepts/cli',
        'concepts/tools',
        //'concepts/servers',
        'concepts/advanced',
    ],
    },
    {
      type: 'category',
      collapsed: false,
      label: 'Tools',
      items: ['tools/elevenlabs']//['tools/perplexity', 'tools/elevenlabs', 'tools/twilio', 'tools/gmail', 'tools/office', 'tools/fal'],
    },
    // {
    //   type: 'category',
    //   label: 'How to Guides',
    //   items: ['guides/memory', 'guides/tools', 'guides/subflow', 'guides/streaming', 'guides/saver'],
    // },
    // {
    //   type: 'category',
    //   label: 'Advanced Features',
    //   items: ['advanced/index', 'advanced/link', 'advanced/data_value'],
    // }
  ],
};

export default sidebars;
