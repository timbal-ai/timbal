// @ts-check

/**
 * Sidebar configuration for Timbal AI v2 documentation
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docsSidebar: [
    'introduction',
    {
      type: 'category',
      collapsed: false,
      label: 'Getting Started',
      items: [
        'getting_started/installation',
        'getting_started/quickstart',
        'getting_started/cursor',
        'getting_started/model_capabilities',
      ],
    },
    {
      type: 'category',
      collapsed: false,
      label: 'Core Concepts',
      items: [
        'core_concepts/runnables',
        'core_concepts/events',
        'core_concepts/context',
      ],
    },
    {
      type: 'category',
      label: 'Agents',
      collapsed: false,
      items: [
        'agents/index',
        'agents/dynamic_agents',
        'agents/rewind',
        'agents/tools',
        'agents/voice',
        'agents/evals',
        'agents/structured_output',
        'agents/advanced',
      ],
    },
    {
      type: 'category',
      label: 'Workflows',
      collapsed: false,
      items: [
        'workflows/index',
        'workflows/context',
        'workflows/control_flow',
        'workflows/integrating_llms',
      ],
    },
  ],
};

module.exports = sidebars;
