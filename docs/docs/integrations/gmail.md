---
title: Gmail
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# Gmail Integration

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Send, search, and manage emails programmatically with Gmail API integration and OAuth authentication.
</h2>

---

Timbal integrates with Gmail to enable sending, searching, and managing emails programmatically. 

This integration allows you to send emails, create drafts, search messages, and handle attachments directly from your Timbal workflows.

## Prerequisites

Before using the Gmail integration, you'll need:

1. A Google account with Gmail access
2. A Google Cloud project with the Gmail API enabled
3. OAuth 2.0 credentials (see [Google Cloud Console](https://console.cloud.google.com))
4. Download your credentials as `gmail_credentials.json` and place it in the appropriate directory
5. Authenticate for the first time to generate `token.json` (see below)

## Authentication Setup

The first time you use the integration, you'll be prompted to authenticate:
<CodeBlock language="bash" code ={`uv run authenticate.py`}/>
This will open a browser window for Google sign-in and create a `token.json` file for future use.

## Installation

Install the requirements by doing:
<CodeBlock language="bash" code ={`uv add timbal[steps-gmail]`}/>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Send Email</strong></span>

### Description
The **send_message** step allows you to send an email (with optional CC, BCC, and attachments) from your authenticated Gmail account.

### Example
<CodeBlock language="python" code ={`from timbal.steps.gmail.messages import send_message

await send_message(
    to=["recipient@example.com"],
    subject="Hello from Timbal!",
    body="This is a test email."
)`}/>

### Parameters

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>to</code></td>
      <td><code>list[str]</code></td>
      <td>Email addresses of the recipients</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>body</code></td>
      <td><code>str</code></td>
      <td>The body of the email</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>subject</code></td>
      <td><code>str</code></td>
      <td>The subject of the email</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>cc</code></td>
      <td><code>list[str]</code></td>
      <td>Email addresses to include in CC</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>bcc</code></td>
      <td><code>list[str]</code></td>
      <td>Email addresses to include in BCC</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>attachment</code></td>
      <td><code>str</code></td>
      <td>Path to a file to attach</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Create Draft</strong></span>

### Description
The **create_draft_message** step allows you to create a draft email (with optional CC, BCC, and attachments).

### Example
<CodeBlock language="python" code ={`from timbal.steps.gmail.messages import create_draft_message

await create_draft_message(
    to=["recipient@example.com"],
    subject="Draft Subject",
    body="Draft content."
)`}/>

### Parameters

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>to</code></td>
      <td><code>list[str]</code></td>
      <td>Email addresses of the recipients</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>body</code></td>
      <td><code>str</code></td>
      <td>The body of the email</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>subject</code></td>
      <td><code>str</code></td>
      <td>The subject of the email</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>cc</code></td>
      <td><code>list[str]</code></td>
      <td>Email addresses to include in CC</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>bcc</code></td>
      <td><code>list[str]</code></td>
      <td>Email addresses to include in BCC</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>attachment</code></td>
      <td><code>str</code></td>
      <td>Path to a file to attach</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Search Messages/Threads</strong></span>

### Description
The **search** step allows you to search Gmail messages or threads using Gmail's search syntax.

### Example
<CodeBlock language="python" code ={`from timbal.steps.gmail.messages import search

results = await search(query="from:someone@example.com", resource="messages", max_results=5)`}/>

### Parameters

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>query</code></td>
      <td><code>str</code></td>
      <td>Search query using Gmail's search syntax</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>resource</code></td>
      <td><code>str</code></td>
      <td>Type of resource to search: "messages" or "threads"</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>max_results</code></td>
      <td><code>int</code></td>
      <td>Maximum number of results to return</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Get Message/Thread</strong></span>

### Description
The **get_message** and **get_thread** steps allow you to retrieve a specific email or thread by ID.

### Example
<CodeBlock language="python" code ={`from timbal.steps.gmail.messages import get_message, get_thread

retrieved_message = await get_message(message_id="your-message-id")
retrieved_thread = await get_thread(thread_id="your-thread-id")`}/>

### Parameters (get_message)

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>message_id</code></td>
      <td><code>str</code></td>
      <td>The ID of the message to get</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>message_format</code></td>
      <td><code>str</code></td>
      <td>Format: "minimal", "full", "raw", or "metadata" (default: "full")</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

### Parameters (get_thread)

<Table className="wider-table">
  <colgroup>
    <col style={{width: "15%"}} />
    <col style={{width: "10%"}} />
    <col style={{width: "60%"}} />
    <col style={{width: "15%"}} />
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Required</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>thread_id</code></td>
      <td><code>str</code></td>
      <td>The ID of the thread to get</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>message_format</code></td>
      <td><code>str</code></td>
      <td>Format: "minimal", "full", "raw", or "metadata" (default: "full")</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## Agent Integration Example

<CodeBlock language="python" code ={`from timbal.steps.gmail.messages import send_message
from timbal import Agent

agent = Agent(
    tools=[send_message]
)

response = await agent.complete(
    prompt={
        "to": ["recipient@example.com"],
        "subject": "Hello from Agent!",
        "body": "This is a test email sent by Agent."
    }
)`}/>

## Notes
- Make sure your Google Cloud project is properly configured and credentials are in place.
- The first authentication will prompt a browser sign-in and create a `token.json` file for future use.
- For more advanced usage, see the [Gmail API documentation](https://developers.google.com/gmail/api/guides).