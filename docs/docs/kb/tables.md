---
title: Tables
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# Tables

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Create and manage structured data tables with defined schemas, data types, and constraints.
</h2>

---

## What is a Table?

A **Table** in a Knowledge Base is a structured container for your data with:

- **Defined columns** with specific data types
- **Constraints** like nullable, unique, and default values  
- **Comments** for documentation
- **PostgreSQL compatibility** for familiar SQL operations

## Creating Tables

### Basic Table Creation

Creates a new structured table in a knowledge base - Defines a table schema with columns, data types, constraints, and documentation.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import create_table, Column

# Create a simple users table
await create_table(
        org_id="your-org-id",
        kb_id="your-kb-id",
        name="Users",
        columns=[
            Column(
                name="id",
                data_type="int", 
                is_nullable=False,
                is_unique=True,
                is_primary=True
            ),
            Column(
                name="email",
                data_type="text",
                is_nullable=False,
                is_unique=True,
                is_primary=False
            ),
            Column(
                name="name", 
                data_type="text",
                is_nullable=False,
                is_unique=False,
                is_primary=False
            ),
            Column(
                name="created_at",
                data_type="timestamp",
                is_nullable=False,
                is_unique=False,
                default_value="NOW()",
                is_primary=False
            )
        ],
        comment="Table for storing user information"
        )`} />

### Resulting Table Schema

The code above creates a table with the following structure:

<div style={{ display: 'flex', justifyContent: 'center', gap: '20px', alignItems: 'center' }}>
  <img src="/img/legend.png" alt="Schema Icons Legend" style={{ maxWidth: '350px', height: 'auto' }} />
  <img src="/img/create_table.png" alt="Users Table Schema" style={{ maxWidth: '280px', height: 'auto' }} />
</div>

### Column Properties

Each column can have the following properties:

<Table className="wider-table">
  <colgroup>
    <col style={{width: "20%"}} />
    <col style={{width: "15%"}} />
    <col style={{width: "50%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Property</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td><code>str</code></td>
      <td>The name of the column</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>data_type</code></td>
      <td><code>str</code></td>
      <td>PostgreSQL data type (text, integer, decimal, boolean, timestamp, etc.)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>is_nullable</code></td>
      <td><code>bool</code></td>
      <td>Whether the column can contain NULL values</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>is_unique</code></td>
      <td><code>bool</code></td>
      <td>Whether the column values must be unique</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>default_value</code></td>
      <td><code>str</code></td>
      <td>Default value for the column (can use SQL functions like NOW())</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>comment</code></td>
      <td><code>str</code></td>
      <td>Description or comment about the column</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

### Supported Data Types

The following PostgreSQL data types are supported:

#### Integer Types
- **`int`** - 32-bit integer
- **`smallint`** - 16-bit integer  
- **`bigint`** - 64-bit integer

#### Text Types
- **`text`** - Variable-length character strings
- **`varchar(n)`** - Variable character with length limit (e.g., `varchar(255)`)
- **`char(n)`** - Fixed-length character (e.g., `char(10)`)
- **`ltree`** - Hierarchical tree-like structures

#### Boolean Type
- **`boolean`** - True/false values

#### Date/Time Types
- **`date`** - Date only
- **`timestamp`** - Date and time (without timezone)
- **`timestamptz`** - Date and time with timezone

#### JSON Types
- **`json`** - JSON data
- **`jsonb`** - Binary JSON data (more efficient)

#### Numeric Types
- **`real`** - 32-bit floating point
- **`double_precision`** - 64-bit floating point
- **`numeric(p,s)`** - Decimal with precision and scale (e.g., `numeric(10,2)`)




## Importing Data

### Import from Records

Takes a list of dictionaries and inserts them as rows in the specified table.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import import_records

# Import data as a list of dictionaries
await import_records(
    org_id="your-org-id",
    kb_id="your-kb-id", 
    table_name="Users",
    records=[
        {
            "id": 1,
            "email": "alice@example.com",
            "name": "Alice Johnson"
        },
        {
            "id": 2, 
            "email": "bob@example.com",
            "name": "Bob Smith"
        },
        {
            "id": 3,
            "email": "carol@example.com", 
            "name": "Carol Davis"
        }
    ]
)`} />

:::warning
**Table Structure**: All fields from the first record determine the table structure. You cannot have different field sets per record - if the first record only has `id`, subsequent records with `id` and `email` will only show the `id` field.
:::

This is what you'll see in the platform when importing records to your table.

<div style={{ display: 'flex', justifyContent: 'center', gap: '20px', alignItems: 'center' }}>
<img src="/img/import_records.png" alt="Import Records" style={{ maxWidth: '780px', height: 'auto' }} />
</div>

### Import from CSV

Reads CSV files and loads the data into existing tables with schema validation.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import import_csv

# Import data from a CSV file
await import_csv(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users", 
    csv_path="./users_data.csv",
    mode="overwrite"  # or "append"
)`} />

The CSV file should have column headers that match your table schema:

<CodeBlock language="csv" code={`id,email,name
1,alice@example.com,Alice Johnson
2,bob@example.com,Bob Smith  
3,carol@example.com,Carol Davis`} />

## Querying Data

### SQL Queries

Executes SQL queries against knowledge base tables.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import query

# Basic SELECT query
results = await query(
    org_id="your-org-id",
    kb_id="your-kb-id",
    sql='SELECT * FROM "Users" WHERE name LIKE \\'%Alice%\\''
)

# Count records
count_result = await query(
    org_id="your-org-id", 
    kb_id="your-kb-id",
    sql='SELECT COUNT(*) as total_users FROM "Users"'
)

# Join tables (if you have multiple tables)
results = await query(
    org_id="your-org-id",
    kb_id="your-kb-id", 
    sql='''
        SELECT u.name, p.title 
        FROM "Users" u 
        JOIN "Posts" p ON u.id = p.user_id
        WHERE u.name = \\'Alice Johnson\\'
    '''
)`} />

### Important SQL Notes

- **Table names are case-sensitive** - Use double quotes around table names: `"Users"`
- **PostgreSQL syntax** - Use PostgreSQL-compatible SQL
- **Escaping quotes** - Use `\\'` for single quotes in strings

### Semantic Search

Performs semantic search on table data using embeddings, you have to specify which is the embedded column that you wanna search on.

:::warning
For semantic search capabilities, you'll need to create embeddings first (see [Embeddings](/kb/embeddings))
:::

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import search_table

# Search using natural language
results = await search_table(
    org_id="your-org-id",
    kb_id="your-kb-id",
    name="Documents", 
    query="artificial intelligence and machine learning",
    embedding_names=["content_embeddings"],
    limit=10,
    offset=0
)`} />

## Managing Tables

### List All Tables

Returns metadata for all tables in the specified knowledge base.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import get_tables

# Get all tables in the knowledge base
tables = await get_tables(
    org_id="your-org-id",
    kb_id="your-kb-id"
)

for table in tables:
    print(f"Table: {table.name}")
    print(f"Comment: {table.comment}")
    print(f"Columns: {len(table.columns)}")
    print("---")`} />

### Get Table Details

Retrieves the complete definition of a specific table - Returns table metadata including column definitions, constraints, and comments.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import get_table

# Get detailed information about a specific table
table = await get_table(
    org_id="your-org-id",
    kb_id="your-kb-id",
    name="Users"
)

print(f"Table name: {table.name}")
print(f"Columns:")
for column in table.columns:
    print(f"  - {column.name}: {column.data_type} (nullable: {column.is_nullable})")`} />

### Get Table SQL Definition

Returns the exact SQL definition that would recreate the table.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import get_table_sql

# Get the CREATE TABLE statement
sql_definition = await get_table_sql(
    org_id="your-org-id", 
    kb_id="your-kb-id",
    name="Users"
)

print(sql_definition)
# Output: CREATE TABLE "Users" ("id" INT NOT NULL, "email" TEXT NOT NULL, ... CONSTRAINT "Users_email_key" UNIQUE (email), ...)`} />

### Get All Tables SQL Definition

Returns the SQL definitions for all tables in the knowledge base.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import get_tables_sql

# Get the CREATE TABLE statements for all tables
sql_definitions = await get_tables_sql(
    org_id="your-org-id", 
    kb_id="your-kb-id"
)

for sql in sql_definitions:
    print(sql)
    print("---")`} />

### Delete Table

Completely removes the table structure and data from the knowledge base.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import delete_table

# Delete a table (be careful!)
await delete_table(
    org_id="your-org-id",
    kb_id="your-kb-id", 
    name="Users",
    cascade=True  # Also delete associated indexes and embeddings
)`} />

## Modifying Tables

### Add Column

Adds a new column to an existing table.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import add_column, Column

# Add a new column to an existing table
await add_column(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    column=Column(
        name="phone_number",
        data_type="varchar(20)",
        is_nullable=True,
        is_unique=False,
        is_primary=False,
        comment="User's phone number"
    )
)`} />

### Drop Column

Removes a column from an existing table.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import drop_column

# Remove a column from the table
await drop_column(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    name="phone_number",
    cascade=True  # Also drop dependent objects
)`} />

### Rename Column

Changes the name of an existing column.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import rename_column

# Rename a column
await rename_column(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    name="name",
    new_name="full_name"
)`} />

### Rename Table

Changes the name of an existing table.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import rename_table

# Rename the entire table
await rename_table(
    org_id="your-org-id",
    kb_id="your-kb-id",
    name="Users",
    new_name="Customers"
)`} />

## Adding Constraints

### Add Foreign Key

Creates a foreign key relationship between tables.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import add_fk

# Add a foreign key constraint
await add_fk(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Posts",
    column_names=["customer_id"],
    fk_table_name="Users",
    fk_column_names=["id"],
    name="fk_posts_customer_id",
    on_delete_action="CASCADE",
    on_update_action="NO ACTION"
)`} />

This creates a relationship between the two tables as shown:

<div style={{ display: 'flex', justifyContent: 'center', gap: '20px', alignItems: 'center' }}>
  <img src="/img/add_fk.png" alt="Foreign Key Relationship" style={{ maxWidth: '660px', height: 'auto' }} />
</div>

<br/>

What happens with this constraint:

- **Data integrity**: You can only insert `customer_id` values in Posts that exist as `id` values in Users
- **Cascade delete**: When a User is deleted, all their Posts are automatically deleted too
- **Referential consistency**: The database ensures the relationship between tables is always valid
- **Query optimization**: The database can optimize joins between these tables more efficiently

### Add Check Constraint

Adds a validation rule to ensure data meets specific conditions.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import add_check

# Add a check constraint
await add_check(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    name="check_email_format",
    expression="email LIKE '%@%'"
)`} />

<br/>

What happens with this constraint:

- **Data validation**: Every insert or update must satisfy the check condition
- **Automatic rejection**: Records that don't meet the criteria are automatically rejected

### Add Unique Constraint

Ensures that values in specified columns are unique across rows.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import add_unique

# Add a unique constraint across multiple columns
await add_unique(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    name="unique_email_domain",
    columns=["email", "domain"]
)`} />

<br/>

What happens with this constraint:

- **Uniqueness enforcement**: No two rows can have the same combination of values in the specified columns
- **Composite uniqueness**: When multiple columns are specified, the combination must be unique (individual columns can still have duplicates)

### Drop Constraint

Removes any type of constraint from a table.

<CodeBlock language="python" code={`from timbal.steps.timbal.tables import drop_constraint

# Remove a constraint
await drop_constraint(
    org_id="your-org-id",
    kb_id="your-kb-id",
    table_name="Users",
    name="check_email_format",
    cascade=True # Also drop dependent objects
)`} />