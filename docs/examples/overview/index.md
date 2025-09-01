---
title: Overview
sidebar: examples
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/categories.module.css';

# Timbal Examples

A collection of examples showcasing how to use **Timbal** to automate AI workflows.

<div className={styles.categoriesGrid}>
  <Link className={styles.categoryBox} to="/examples/agents">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>Agents</strong></span></h2>
    </div>
    <p>AI agents and automation solutions</p>
  </Link>

  <Link className={styles.categoryBox} to="/examples/workflows">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>Workflows</strong></span></h2>
    </div>
    <p>Multi-step process automation</p>
  </Link>

  <Link className={styles.categoryBox} to="/examples/tools">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>Tools</strong></span></h2>
    </div>
    <p>External integrations and APIs</p>
  </Link>

  <Link className={styles.categoryBox} to="/examples/memory">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>Memory</strong></span></h2>
    </div>
    <p>Context and conversation management</p>
  </Link>

  <Link className={styles.categoryBox} to="/examples/rag">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>RAG</strong></span></h2>
    </div>
    <p>Retrieval augmented generation</p>
  </Link>

  <Link className={styles.categoryBox} to="/examples/evals">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>Evals</strong></span></h2>
    </div>
    <p>Evaluation and testing frameworks</p>
  </Link>

  <Link className={styles.categoryBox} to="/examples/voice">
    <div className={styles.categoryTitleRow}>
      <h2> <span style={{color: 'var(--timbal-purple)'}}><strong>Voice</strong></span></h2>
    </div>
    <p>Voice-based interactions</p>
  </Link>


</div>