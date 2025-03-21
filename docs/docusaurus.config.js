// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Timbal AI',
  tagline: 'A strongly opinionated framework for building and orchestrating agentic AI applications.',
  favicon: 'img/favicon.png',

  // Set the production url of your site here
  url: 'https://timbal-ai.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/timbal/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'timbal-ai', // Usually your GitHub org/user name.
  projectName: 'timbal', // Usually your repo name.
  trailingSlash: false,
  deploymentBranch: 'gh-pages',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        logo: {
          alt: 'Timbal Logo',
          src: 'img/LogoBlack.svg',
          srcDark: 'img/LogoWhite.svg',
          width: '100px',  
          height: 'auto',
        },
        items: [
          { to: '/docs/get-started', label: 'Docs', position: 'left' },
          //{ to: '/docs/examples', label: 'Examples', position: 'left' },
          {
            href: 'https://github.com/timbal-ai/timbal',
            label: 'GitHub',
            position: 'right',
          }
        ]
      },
      footer: {
        style: 'light',
        links: [
          {
            title: 'Follow us',
            items: [
              {
                label: 'Website',
                href: 'https://timbal.ai',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/timbal-ai/timbal',
              },
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/company/timbal-ai/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Timbal AI. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
     }),
};

export default config;
