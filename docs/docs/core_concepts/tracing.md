---
title: Tracing & Observability
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';


# Tracing & Observability

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Comprehensive execution tracing with input/output/error/timing capture for complete observability
</h2>

---

---

## What are Traces?

Timbal provides comprehensive execution tracing that captures every aspect of runnable execution, **from input parameters to final outputs, including timing data, error states, and usage metrics**. A <span style={{color: 'var(--timbal-purple)'}}><strong>Trace</strong></span> is the core data structure that captures execution information for every runnable execution, providing a complete audit trail of what happened. This tracing system enables complete observability into your application's behavior and performance.


### Events vs. Traces:
Output Events and Traces have a similar data, but serve different purposes. Events are in fact the source data that gets processed into persistent Trace records.

- <span style={{color: 'var(--timbal-purple)'}}><strong>Events</strong></span> are real-time notifications that stream during execution. They provide immediate feedback about what's happening (start, progress chunks, completion) but are not stored permanently. Events are consumed as they're generated and are ideal for real-time monitoring, progress tracking, and streaming responses.

- <span style={{color: 'var(--timbal-purple)'}}><strong>Traces</strong></span> are persistent records that capture the complete execution context. They contain detailed execution metadata, timing information, resource usage, and complete input/output snapshots.

**Memory Management and Persistence**: Traces persistence enable memory management by tracking resource usage patterns, maintaining execution history for debugging, and supporting long-term performance analysis and audit trails.
