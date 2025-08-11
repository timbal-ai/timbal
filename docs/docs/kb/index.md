---
title: Overview
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Knowledge Bases

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Store, organize, and search through your data with structured tables, semantic search, and powerful query capabilities.
</h2>

---

:::info Platform Requirement

**Knowledge Bases are only available in the Timbal Platform.** 

You must create your Knowledge Base from the [Timbal Platform](https://platform.timbal.ai), and a valid **Timbal API token** is required to access it.

Throughout this documentation, any references to **org ID** and **kb ID** refer to the identifiers provided by the Timbal Platform for your organization and knowledge base.

:::


## What is a Knowledge Base?

Think of a **Knowledge Base** as a smart database that combines the best of both worlds:

### Traditional Database Capabilities
- Store structured data in organized tables (like spreadsheets)
- Query data with precise SQL commands
- Maintain data relationships and integrity
- Import data from files or add it programmatically

### AI-Powered Intelligence
- Search using natural language ("Find documents about API integration")
- Understand context and meaning, not just exact words
- Find similar content automatically
- Learn from your data patterns

## The Three Building Blocks

Knowledge Bases are built on three core components that work together:

### 1. Tables - Your Data Foundation
Think of tables as organized filing cabinets. Each table is like a cabinet with labeled drawers (columns) where you store specific types of information.

**Example**: A "Products" table might have drawers for:
- Product name
- Description  
- Price
- Category
- Stock quantity

### 2. Embeddings - The AI Brain
Embeddings are like giving your data a "brain" that understands meaning. They convert text into mathematical representations that capture the essence of what the text is about.

**How it works**: 
- "Laptop computer" and "portable PC" get similar mathematical representations
- "Customer support" and "help desk" are recognized as related concepts
- Natural language queries find relevant content even without exact matches

### 3. Indexes - The Speed Boosters
Indexes are like the index at the back of a book - they help you find information quickly without reading every page. They create fast pathways to your data.

**Types of speed boosts**:
- **B-tree indexes**: Fast lookups for names, dates, numbers
- **Hash indexes**: Lightning-fast exact matches
- **GIN indexes**: Powerful text search capabilities
- **Composite indexes**: Optimized for complex queries

## How Knowledge Bases Transform Your Data

### Before Knowledge Bases
<CodeBlock language="bash" code={`User: "I need help with password reset"
System: ❌ No results found (exact match only)`} />

### With Knowledge Bases
<CodeBlock language="bash" code={`User: "I forgot my login credentials"
System: ✅ Found: "How to reset your password" 
        ✅ Found: "Account recovery process"
        ✅ Found: "Troubleshooting login issues"`} />

## The Knowledge Base Workflow

<div style={{
  display: 'flex',
  flexDirection: 'column',
  gap: '2rem',
  marginTop: '2rem',
  marginBottom: '2rem'
}}>

<div style={{
  display: 'flex',
  alignItems: 'center',
  gap: '1.5rem',
  padding: '1.5rem',
  background: 'var(--ifm-background-color)',
  borderRadius: '16px',
  border: '1px solid var(--ifm-color-primary)',
  boxShadow: '0 4px 12px rgba(80, 18, 203, 0.1)'
}}>
  <div style={{
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: '#c6b8ff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#5012cb',
    fontWeight: 'bold',
    fontSize: '1.2rem',
    boxShadow: '0 4px 12px rgba(198, 184, 255, 0.3)'
  }}>
    1
  </div>
  <div>
    <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--ifm-color-primary)', fontSize: '1.1rem', fontWeight: '600' }}>
      Design Your Data Structure
    </h4>
    <p style={{ margin: 0, color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
      Plan what information you want to store and how it should be organized.
    </p>
  </div>
</div>

<div style={{
  display: 'flex',
  alignItems: 'center',
  gap: '1.5rem',
  padding: '1.5rem',
  background: 'var(--ifm-background-color)',
  borderRadius: '16px',
  border: '1px solid var(--ifm-color-primary)',
  boxShadow: '0 4px 12px rgba(80, 18, 203, 0.1)'
}}>
  <div style={{
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: '#c6b8ff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#5012cb',
    fontWeight: 'bold',
    fontSize: '1.2rem',
    boxShadow: '0 4px 12px rgba(198, 184, 255, 0.3)'
  }}>
    2
  </div>
  <div>
    <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--ifm-color-primary)', fontSize: '1.1rem', fontWeight: '600' }}>
      Create Tables
    </h4>
    <p style={{ margin: 0, color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
      Set up the structure for your data with appropriate columns and data types.
    </p>
  </div>
</div>

<div style={{
  display: 'flex',
  alignItems: 'center',
  gap: '1.5rem',
  padding: '1.5rem',
  background: 'var(--ifm-background-color)',
  borderRadius: '16px',
  border: '1px solid var(--ifm-color-primary)',
  boxShadow: '0 4px 12px rgba(80, 18, 203, 0.1)'
}}>
  <div style={{
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: '#c6b8ff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#5012cb',
    fontWeight: 'bold',
    fontSize: '1.2rem',
    boxShadow: '0 4px 12px rgba(198, 184, 255, 0.3)'
  }}>
    3
  </div>
  <div>
    <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--ifm-color-primary)', fontSize: '1.1rem', fontWeight: '600' }}>
      Import Your Data
    </h4>
    <p style={{ margin: 0, color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
      Add your information from files, databases, or programmatically.
    </p>
  </div>
</div>

<div style={{
  display: 'flex',
  alignItems: 'center',
  gap: '1.5rem',
  padding: '1.5rem',
  background: 'var(--ifm-background-color)',
  borderRadius: '16px',
  border: '1px solid var(--ifm-color-primary)',
  boxShadow: '0 4px 12px rgba(80, 18, 203, 0.1)'
}}>
  <div style={{
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: '#c6b8ff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#5012cb',
    fontWeight: 'bold',
    fontSize: '1.2rem',
    boxShadow: '0 4px 12px rgba(198, 184, 255, 0.3)'
  }}>
    4
  </div>
  <div>
    <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--ifm-color-primary)', fontSize: '1.1rem', fontWeight: '600' }}>
      Enable AI Search
    </h4>
    <p style={{ margin: 0, color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
      Create embeddings on text columns to enable semantic search capabilities.
    </p>
  </div>
</div>

<div style={{
  display: 'flex',
  alignItems: 'center',
  gap: '1.5rem',
  padding: '1.5rem',
  background: 'var(--ifm-background-color)',
  borderRadius: '16px',
  border: '1px solid var(--ifm-color-primary)',
  boxShadow: '0 4px 12px rgba(80, 18, 203, 0.1)'
}}>
  <div style={{
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: '#c6b8ff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#5012cb',
    fontWeight: 'bold',
    fontSize: '1.2rem',
    boxShadow: '0 4px 12px rgba(198, 184, 255, 0.3)'
  }}>
    5
  </div>
  <div>
    <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--ifm-color-primary)', fontSize: '1.1rem', fontWeight: '600' }}>
      Optimize Performance
    </h4>
    <p style={{ margin: 0, color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
      Add indexes to ensure fast queries as your data grows.
    </p>
  </div>
</div>

<div style={{
  display: 'flex',
  alignItems: 'center',
  gap: '1.5rem',
  padding: '1.5rem',
  background: 'var(--ifm-background-color)',
  borderRadius: '16px',
  border: '1px solid var(--ifm-color-primary)',
  boxShadow: '0 4px 12px rgba(80, 18, 203, 0.1)'
}}>
  <div style={{
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: '#c6b8ff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#5012cb',
    fontWeight: 'bold',
    fontSize: '1.2rem',
    boxShadow: '0 4px 12px rgba(198, 184, 255, 0.3)'
  }}>
    6
  </div>
  <div>
    <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--ifm-color-primary)', fontSize: '1.1rem', fontWeight: '600' }}>
      Search and Discover
    </h4>
    <p style={{ margin: 0, color: 'var(--ifm-font-color-base)', lineHeight: '1.5' }}>
    </p>
  </div>
</div>

</div>

---

## Getting Started

Ready to build your first Knowledge Base? Start with these guides:

- **[Tables](/kb/tables)** - Learn how to create and organize your data
- **[Embeddings](/kb/embeddings)** - Enable AI-powered semantic search
- **[Indexes](/kb/indexes)** - Optimize performance for fast queries

---
