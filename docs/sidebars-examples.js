// @ts-check

/**
 * Sidebar configuration for Timbal AI v2 documentation
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
    docsSidebar: [
      'overview/index',
      {
        type: 'category',
        collapsed: false,
        label: 'Agents',
        items: [
          'agents/calling_agents',
          'agents/agent_system_prompt',
          'agents/adding_tool',
          'agents/adding_workflow',
          'agents/supervisor_agent',
          'agents/image_analysis',
          'agents/using_voice',
          'agents/dynamic_context',
        ],
      },
      {
        type: 'category',
        label: 'Workflows',
        collapsed: false,
        items: [
          'workflows/running_workflows',
          'workflows/sequential_steps',
          'workflows/parallel_steps',
          'workflows/conditional_branching',
          'workflows/array_input',
          'workflows/calling_agent',
          'workflows/agent_step',
          'workflows/tool_step',
          'workflows/human_loop',

        ],
      },
    //   {
    //     type: 'category',
    //     label: 'Workflows',
    //     collapsed: false,
    //     items: [
    //       'workflows_v2/index',
    //     ],
    //   },
    //   {
    //     type: 'category',
    //     label: 'Advanced',
    //     collapsed: false,
    //     items: [
    //       'advanced_v2/tracing',
    //     ],
    //   },
    //   {
    //     type: 'category',
    //     collapsed: false,
    //     label: 'Integrations',
    //     items: [
    //       'integrations_v2/index',
    //       'integrations_v2/elevenlabs',
    //       'integrations_v2/perplexity',
    //       'integrations_v2/twilio',
    //       'integrations_v2/gmail',
    //       'integrations_v2/sharepoint',
    //       'integrations_v2/fal',
    //     ],
    //   },
    ],
  };
  
  module.exports = sidebars;
  