---
title: Indexes
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# Indexes

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Optimize query performance and enable advanced search capabilities with database indexes on your table columns.
</h2>

---

## What are Indexes?

**Indexes** are data structures that improve query performance by creating optimized access paths to your data:

- **Speed up queries** - Find data faster instead of scanning entire tables
- **Enable advanced searches** - Support complex search patterns and operations
- **Optimize specific use cases** - Different index types for different needs
- **Trade-off storage for speed** - Use extra storage to achieve faster queries

## Types of Indexes

Timbal supports several PostgreSQL index types, each optimized for different use cases:

<div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', margin: '30px 0'}}>

<div style={{
  border: '2px solid var(--ifm-color-emphasis-300)',
  borderRadius: '12px',
  padding: '24px',
  backgroundColor: 'var(--ifm-background-surface-color)',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  transition: 'all 0.2s ease'
}}>
<h3 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1.3rem'}}>🌳 B-tree (btree)</h3>
<p style={{margin: '0 0 8px 0', fontWeight: '600', color: 'var(--ifm-font-color-base)'}}>Default and Most Common</p>
<p style={{margin: '0', color: 'var(--ifm-color-emphasis-700)', lineHeight: '1.5'}}>Best for equality and range queries on sortable data.</p>
</div>

<div style={{
  border: '2px solid var(--ifm-color-emphasis-300)',
  borderRadius: '12px',
  padding: '24px',
  backgroundColor: 'var(--ifm-background-surface-color)',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  transition: 'all 0.2s ease'
}}>
<h3 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1.3rem'}}>🔍 Hash</h3>
<p style={{margin: '0 0 8px 0', fontWeight: '600', color: 'var(--ifm-font-color-base)'}}>Exact Matches Only</p>
<p style={{margin: '0', color: 'var(--ifm-color-emphasis-700)', lineHeight: '1.5'}}>Optimized for exact equality comparisons.</p>
</div>

<div style={{
  border: '2px solid var(--ifm-color-emphasis-300)',
  borderRadius: '12px',
  padding: '24px',
  backgroundColor: 'var(--ifm-background-surface-color)',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  transition: 'all 0.2s ease'
}}>
<h3 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1.3rem'}}>📝 GIN (Generalized Inverted Index)</h3>
<p style={{margin: '0 0 8px 0', fontWeight: '600', color: 'var(--ifm-font-color-base)'}}>Full-Text Search</p>
<p style={{margin: '0', color: 'var(--ifm-color-emphasis-700)', lineHeight: '1.5'}}>Perfect for searching within text, arrays, and JSON data.</p>
</div>

<div style={{
  border: '2px solid var(--ifm-color-emphasis-300)',
  borderRadius: '12px',
  padding: '24px',
  backgroundColor: 'var(--ifm-background-surface-color)',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  transition: 'all 0.2s ease'
}}>
<h3 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1.3rem'}}>🌐 GiST (Generalized Search Tree)</h3>
<p style={{margin: '0 0 8px 0', fontWeight: '600', color: 'var(--ifm-font-color-base)'}}>Geometric and Complex Data</p>
<p style={{margin: '0', color: 'var(--ifm-color-emphasis-700)', lineHeight: '1.5'}}>Good for geometric data, full-text search, and nearest-neighbor searches.</p>
</div>

<div style={{
  border: '2px solid var(--ifm-color-emphasis-300)',
  borderRadius: '12px',
  padding: '24px',
  backgroundColor: 'var(--ifm-background-surface-color)',
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  transition: 'all 0.2s ease'
}}>
<h3 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1.3rem'}}>📊 BRIN (Block Range Index)</h3>
<p style={{margin: '0 0 8px 0', fontWeight: '600', color: 'var(--ifm-font-color-base)'}}>Large Tables with Natural Order</p>
<p style={{margin: '0', color: 'var(--ifm-color-emphasis-700)', lineHeight: '1.5'}}>Efficient for very large tables where data has natural clustering.</p>
</div>

</div>

## Creating Indexes

### Basic Index Creation

<CodeBlock language="python" code={`from timbal.steps.timbal.indexes import create_index

# Create a basic B-tree index on a single column
await create_index(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    name="idx_users_email",        # Index name
    column_names=["email"],        # Columns to index
    type="btree",                  # Index type
    is_unique=True                 # Enforce uniqueness
)`} />


:::warning IMPORTANT!
- **Index names must be unique across your entire knowledge base** - two different tables cannot have indexes with the same name
- **Only create indexes for columns you actually query** - unnecessary indexes waste storage and slow down writes
:::

### Index Types

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "40%"}} />
    <col style={{width: "30%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Type</th>
      <th>Best For</th>
      <th>Example Use Cases</th>
      <th>Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>btree</code></td>
      <td>Equality, ranges, sorting</td>
      <td>User IDs, dates, names, prices</td>
      <td>Excellent</td>
    </tr>
    <tr>
      <td><code>hash</code></td>
      <td>Exact equality only</td>
      <td>Status codes, categories</td>
      <td>Very Fast</td>
    </tr>
    <tr>
      <td><code>gin</code></td>
      <td>Text search, arrays, JSON</td>
      <td>Full-text search, tags, metadata</td>
      <td>Good</td>
    </tr>
    <tr>
      <td><code>gist</code></td>
      <td>Geometric, similarity</td>
      <td>Locations, nearest neighbors</td>
      <td>Good</td>
    </tr>
    <tr>
      <td><code>brin</code></td>
      <td>Large naturally ordered data</td>
      <td>Time series, log data</td>
      <td>Space Efficient</td>
    </tr>
  </tbody>
</Table>

### Multiple Column Indexes

#### Creating Composite Indexes

<CodeBlock language="python" code={`# Create a composite index on multiple columns
await create_index(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Orders",
    name="idx_orders_customer_date",
    column_names=["customer_id", "order_date"],  # Multiple columns
    type="btree",
    is_unique=False
)`} />

**Column Order Matters:** Put the most selective columns first (columns with many unique values) before less selective ones.

<div style={{
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '20px',
  margin: '16px 0'
}}>

<div style={{
  border: '2px solid var(--ifm-color-success)',
  borderRadius: '8px',
  padding: '16px',
  backgroundColor: 'var(--ifm-color-success-contrast-background)'
}}>
<h5 style={{margin: '0 0 8px 0', color: 'var(--ifm-color-success)', fontSize: '1rem'}}>✅ Good Order</h5>
<code style={{fontSize: '0.9rem'}}>["user_id", "status"]</code>
<p style={{margin: '8px 0 0 0', fontSize: '0.9rem', color: 'var(--ifm-color-success)'}}>user_id has many unique values</p>
</div>

<div style={{
  border: '2px solid var(--ifm-color-danger)',
  borderRadius: '8px',
  padding: '16px',
  backgroundColor: 'var(--ifm-color-danger-contrast-background)'
}}>
<h5 style={{margin: '0 0 8px 0', color: 'var(--ifm-color-danger)', fontSize: '1rem'}}>❌ Poor Order</h5>
<code style={{fontSize: '0.9rem'}}>["status", "user_id"]</code>
<p style={{margin: '8px 0 0 0', fontSize: '0.9rem', color: 'var(--ifm-color-danger)'}}>status has few unique values</p>
</div>

</div>

#### Query Optimization & Partial Usage

<div style={{
  border: '1px solid var(--ifm-color-emphasis-300)',
  borderRadius: '8px',
  padding: '20px',
  backgroundColor: 'var(--ifm-background-surface-color)',
  margin: '16px 0'
}}>

<p style={{margin: '0 0 16px 0', fontWeight: '600', color: 'var(--ifm-font-color-base)'}}>
<strong>Rule:</strong> You can use part of a composite index, but only from left to right.
</p>

<div style={{
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '24px',
  margin: '16px 0'
}}>

<div>
<h5 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1rem'}}>✅ Efficient Queries</h5>
<div style={{
  backgroundColor: 'var(--ifm-background-color)',
  border: '1px solid var(--ifm-color-emphasis-200)',
  borderRadius: '6px',
  padding: '12px',
  margin: '8px 0'
}}>
<div style={{fontSize: '0.9rem', fontFamily: 'monospace'}}>
SELECT * FROM Orders<br/>
WHERE customer_id = 123<br/>
AND order_date > '2024-01-01'
</div>
<div style={{fontSize: '0.8rem', color: 'var(--ifm-color-emphasis-600)', marginTop: '4px'}}>Uses both columns</div>
</div>
<div style={{
  backgroundColor: 'var(--ifm-background-color)',
  border: '1px solid var(--ifm-color-emphasis-200)',
  borderRadius: '6px',
  padding: '12px',
  margin: '8px 0'
}}>
<div style={{fontSize: '0.9rem', fontFamily: 'monospace'}}>
SELECT * FROM Orders<br/>
WHERE customer_id = 123
</div>
<div style={{fontSize: '0.8rem', color: 'var(--ifm-color-emphasis-600)', marginTop: '4px'}}>Uses partial index (1st column)</div>
</div>
</div>

<div>
<h5 style={{margin: '0 0 12px 0', color: 'var(--ifm-color-primary)', fontSize: '1rem'}}>❌ Inefficient Queries</h5>
<div style={{
  backgroundColor: 'var(--ifm-background-color)',
  border: '1px solid var(--ifm-color-emphasis-200)',
  borderRadius: '6px',
  padding: '12px',
  margin: '8px 0'
}}>
<div style={{fontSize: '0.9rem', fontFamily: 'monospace'}}>
SELECT * FROM Orders<br/>
WHERE order_date > '2024-01-01'
</div>
<div style={{fontSize: '0.8rem', color: 'var(--ifm-color-emphasis-600)', marginTop: '4px'}}>Skips 1st column</div>
</div>
<div style={{
  backgroundColor: 'var(--ifm-background-color)',
  border: '1px solid var(--ifm-color-emphasis-200)',
  borderRadius: '6px',
  padding: '12px',
  margin: '8px 0'
}}>

<div style={{fontSize: '0.9rem', fontFamily: 'monospace'}}>
SELECT * FROM Orders<br/>
WHERE order_date = '2024-01-01'<br/>
AND customer_id = 123
</div>

<div style={{fontSize: '0.8rem', color: 'var(--ifm-color-emphasis-600)', marginTop: '4px'}}>Wrong column order</div>
</div>
</div>

</div>

> 💡 <b>Tip:</b> Think of it like reading a book - you can start from the beginning and read forward, but you can't skip chapters.

</div>

## Managing Indexes

### List All Indexes

Discovers all indexes in your knowledge base and shows detailed information about each index:

- **name**: What you called the index
- **table**: Which table it's on  
- **type**: B-tree, hash, GIN, etc.
- **columns**: Which columns are indexed
- **is_unique**: Whether it enforces uniqueness
- **definition**: The actual SQL definition

<CodeBlock language="python" code={`from timbal.steps.timbal.indexes import list_indexes

# List all indexes in the knowledge base
indexes = await list_indexes(
    org_id="your-org-id",
    kb_id="your-kb-id"
)

for index in indexes:
    print(f"Index: {index.name}")
    print(f"Table: {index.table}")  
    print(f"Type: {index.type}")
    print(f"Columns: {', '.join(index.columns)}")
    print(f"Unique: {index.is_unique}")
    print(f"Definition: {index.definition}")
    print("---")
`} />

### List Indexes for a Specific Table

You can also filter indexes to show only those for a specific table by adding the `table_name` parameter:

<CodeBlock language="python" code={`from timbal.steps.timbal.indexes import list_indexes

# List indexes for a specific table
table_indexes = await list_indexes(
    org_id="your-org-id",
    kb_id="your-kb-id", 
    table_name="Users"
)
`} />

### Delete Indexes

Removes an index completely from your database.

<CodeBlock language="python" code={`from timbal.steps.timbal.indexes import delete_index

# Delete an index (be careful!)
await delete_index(
    org_id="your-org-id",
    kb_id="your-kb-id",
    name="idx_old_index"
)`} />




---

Indexes are a powerful tool for optimizing your Knowledge Base performance. 

Combined with well-designed [Tables](./tables) and strategic [Embeddings](./embeddings), they ensure your Knowledge Base can handle complex queries efficiently as your data grows.