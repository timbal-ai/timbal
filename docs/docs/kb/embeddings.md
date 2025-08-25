---
title: Embeddings
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# Embeddings

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Enable AI-powered semantic search by converting text data into vector representations that understand meaning and context.
</h2>

---

## What are Embeddings?

**Embeddings** are numerical vector representations of text that capture semantic meaning. When you create an embedding on a column:

- **Text is converted** into high-dimensional vectors
- **Similar content** gets similar vector representations  
- **Semantic search** becomes possible using natural language queries
- **Context is understood** - "car" and "automobile" would be considered similar

## How Embeddings Work

1. **Text Processing**: Your text data is processed by an embedding model
2. **Vector Generation**: Each piece of text becomes a vector (array of numbers)
3. **Storage**: Vectors are stored alongside your original data
4. **Search**: Query text is converted to a vector and compared against stored vectors
5. **Results**: Most similar vectors (and their associated data) are returned

## Creating Embeddings

### Basic Embedding Creation

<CodeBlock language="python" code={`from timbal.steps.timbal.embeddings import create_embedding

# Create an embedding on a text column
await create_embedding(
    org_id="your-org-id",
    kb_id="your-kb-id",
    name="product_descriptions",  # Name for this embedding
    table_name="Products",        # Table containing the text
    column_name="description",    # Column to create embeddings for
    model="text-embedding-3-small",  # Embedding model to use
    with_gin_index=True          # Create index for better performance
)`} />

You can create multiple embeddings on different columns or even the same column with different models.

:::warning IMPORTANT!
- **Embeddings can only be created on text columns** - they convert text into vector representations
- **Embedding names must be unique across your entire knowledge base** - two different tables cannot have embeddings with the same name
- **Only create embeddings on columns you want to search semantically** - unnecessary embeddings waste storage and processing time
:::

### Available Embedding Models

First, check what models are available for your organization:

<CodeBlock language="python" code={`from timbal.steps.timbal.embeddings import list_embedding_models

# Get available embedding models
models = await list_embedding_models(org_id="your-org-id")

print("Available embedding models:")
for model in models:
    print(f"  - {model}")`} />

Common embedding models include:

- **`text-embedding-3-small`** - OpenAI's efficient model, good for most use cases
- **`text-embedding-3-large`** - OpenAI's most powerful model, higher accuracy
- **`text-embedding-ada-002`** - OpenAI's previous generation model
- Additional models may be available depending on your organization setup


## Using Embeddings for Search

### Semantic Search

Once embeddings are created, you can perform semantic searches:

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import search_table

# Search using natural language
results = await search_table(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Articles",
    query="How to improve website performance?",  # Natural language query
    embedding_names=["content_embeddings"],       # Which embeddings to use
    limit=10,
    offset=0
)

# Results are ordered by relevance
for result in results:
    print(f"Title: {result['title']}")
    print("---")`} />


:::warning ATTENTION!
**Searches can only be performed on one column at a time.** You cannot search across multiple embedding columns simultaneously.
:::


## Managing Embeddings

### List Embeddings

<CodeBlock language="python" code={`from timbal.steps.timbal.embeddings import list_embeddings

# List all embeddings in a knowledge base
embeddings = await list_embeddings(
    org_id="your-org-id",
    kb_id="your-kb-id"
)

for embedding in embeddings:
    print(f"Name: {embedding.name}")
    print(f"Table: {embedding.table_name}")
    print(f"Column: {embedding.column_name}")
    print(f"Model: {embedding.model}")
    print(f"Status: {embedding.status}")
    print("---")

# List embeddings for a specific table
table_embeddings = await list_embeddings(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Articles"
)`} />


### Delete Embeddings

<CodeBlock language="python" code={`from timbal.steps.timbal.embeddings import delete_embedding

# Delete an embedding (be careful!)
await delete_embedding(
    org_id="your-org-id",
    kb_id="your-kb-id",
    name="old_embeddings"
)`} />


## Embedding Success Tips

Embeddings aren’t just for plain text! You can create them on any column that helps users find what they need—titles, tags, categories, and more.

- Set <code>with_gin_index=True</code> for large columns
- Create separate embeddings for different search patterns
- Monitor embedding status before searching

> 💡 <b>Tip:</b> The more thoughtfully you choose your columns, the smarter and faster your search will be!
