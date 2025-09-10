// @ts-check

/**
 * Sidebar configuration for Timbal AI v2 documentation
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    'introduction_v2/index',
    {
      type: 'category',
      collapsed: false,
      label: 'Getting Started',
      items: [
        'getting-started_v2/installation',
        'getting-started_v2/quickstart',
        'getting-started_v2/cursor',
        'getting-started_v2/model_capabilities',
      ],
    },
    {
      type: 'category',
      collapsed: false,
      label: 'Core Concepts',
      items: [
        'core_concepts_v2/runnables',
        'core_concepts_v2/events',
        'core_concepts_v2/context',
        'core_concepts_v2/tracing',
      ],
    },
    {
      type: 'category',
      label: 'Agents',
      collapsed: false,
      items: [
        'agents_v2/index',
        'agents_v2/dynamic_agents',
        'agents_v2/rewind',
        'agents_v2/tools',
        'agents_v2/voice',
        'agents_v2/evals',
        'agents_v2/structured_output',
        'agents_v2/advanced',
      ],
    },
    {
      type: 'category',
      label: 'Workflows',
      collapsed: false,
      items: [
        'workflows_v2/index',
        'workflows_v2/control_flow',
        'workflows_v2/context',
        'workflows_v2/integrating_llms',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      collapsed: false,
      items: [
        'advanced_v2/tracing',
      ],
    },
    {
      type: 'category',
      collapsed: false,
      label: 'Integrations',
      items: [
        'integrations_v2/index',
        'integrations_v2/elevenlabs',
        'integrations_v2/perplexity',
        'integrations_v2/twilio',
        'integrations_v2/gmail',
        'integrations_v2/sharepoint',
        'integrations_v2/fal',
      ],
    },
  ],
};

module.exports = sidebars;
