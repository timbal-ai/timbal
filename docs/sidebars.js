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
    'introduction/index',
    {
      type: 'category',
      collapsed: false,
      label: 'Getting Started',
      items: ['getting-started/installation', 'getting-started/quickstart', 'getting-started/cursor', 'getting-started/model_capabilities'],
    },
    {
      type: 'category',
      label: 'Agents',
      collapsed: false,
      items: [
        'agents/index',
        'agents/memory',
        'agents/rewind',
        'agents/tools',
        'agents/voice',
        'agents/evals',
    ],
    },
    {
      type: 'category',
      label: 'Flows',
      collapsed: false,
      items: [
        'flows/index',
        'flows/memory',
        'flows/advanced',
    ],
    },
    {
      type: 'category',
      label: 'State',
      collapsed: false,
      items: [
        'state/index',
    ],
    },
    {
      type: 'category',
      collapsed: false,
      label: 'Integrations',
      items: ['integrations/index', 'integrations/elevenlabs', 'integrations/perplexity', 'integrations/twilio', 'integrations/gmail', 'integrations/sharepoint', 'integrations/fal'],
    }
  ],
  examplesSidebar: [
    {
      type: 'category',
      collapsed: false,
      label: 'Timbal Examples',
      items: ['examples/index', 'examples/healthcare', 'examples/automotive', 'examples/design', 'examples/tourism', 'examples/financials', 'examples/ecommerce', 'examples/industry', 'examples/real-estate', 'examples/education', 'examples/legal', 'examples/transport', 'examples/media'],
    }
  ],
};

export default sidebars;
