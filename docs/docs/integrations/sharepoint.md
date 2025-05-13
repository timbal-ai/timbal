---
title: SharePoint
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';
import Table from '@site/src/components/Table';

# SharePoint Integration

Timbal integrates with SharePoint to enable listing, downloading, and managing files programmatically. This integration allows you to access both personal and shared files across different scopes in your SharePoint environment.

## Prerequisites

Before using the SharePoint integration, you'll need:

1. A [Microsoft 365 tenant](https://portal.azure.com/#view/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade/~/Overview)
2. A registered application in [Azure Portal](https://portal.azure.com/#view/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade/~/RegisteredApps) with:
   - Application (client) ID
   - Directory (tenant) ID
   - Client secret
3. Required permissions configured:
   - Files.Read.All
   - Sites.Read.All
4. A `token_sharepoint.json` file with your credentials

## Authentication Setup

The first time you use the integration, you'll be prompted to authenticate:
<CodeBlock language="bash" code ={`uv run authenticate.py`}/>
This will open a browser window for Microsoft sign-in and create a token file for future use.

## Installation

Install the requirements by doing:
<CodeBlock language="bash" code ={`uv add timbal[steps-sharepoint]`}/>

## <span style={{color: 'var(--timbal-purple)'}}><strong>List Directory</strong></span>

### Description
The **list_directory** step allows you to list files and folders in SharePoint, with support for both personal and shared files across different scopes.

### Example
<CodeBlock language="python" code ={`from timbal.steps.sharepoint.directories import list_directory

# List all files in personal drive
items = await list_directory(scope="my-files")

# List contents of a specific folder
folder_items = await list_directory(
    folder="Projects/2024",
    scope="my-files"
)

# List shared files
shared_items = await list_directory(scope="shared")`}/>

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
      <td><code>folder</code></td>
      <td><code>str</code></td>
      <td>The SharePoint folder path to list contents from</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>scope</code></td>
      <td><code>str</code></td>
      <td>Where to search for files: 'my-files', 'shared', or 'all'</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>output_path</code></td>
      <td><code>str</code></td>
      <td>Path to save the output JSON file</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## <span style={{color: 'var(--timbal-purple)'}}><strong>Download File</strong></span>

### Description
The **download_file** step allows you to download files from SharePoint by searching for them across different scopes.

### Example
<CodeBlock language="python" code ={`from timbal.steps.sharepoint.files import download_file

# Download a file from personal drive
success = await download_file("document.docx")

# Download from a specific folder
success = await download_file(
    "report.pdf",
    folder="Projects/2024"
)

# Download from shared files
success = await download_file(
    "presentation.pptx",
    scope="shared"
)

# Download to specific location
success = await download_file(
    "data.xlsx",
    destination="/path/to/save"
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
      <td><code>file</code></td>
      <td><code>str</code></td>
      <td>Name or ID of the file to find and download</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td><code>destination</code></td>
      <td><code>str</code></td>
      <td>Directory where the file will be downloaded to. If None, uses the current working directory</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>folder</code></td>
      <td><code>str</code></td>
      <td>Optional path to a specific folder to search in</td>
      <td>No</td>
    </tr>
    <tr>
      <td><code>scope</code></td>
      <td><code>str</code></td>
      <td>Search scope ('my-files', 'shared', or 'all'). Defaults to 'my-files'</td>
      <td>No</td>
    </tr>
  </tbody>
</Table>

## Agent Integration Example

<CodeBlock language="python" code ={`from timbal.steps.sharepoint.directories import list_directory
from timbal import Agent

agent = Agent(
    tools=[list_directory]
)

response = await agent.complete(
    prompt={
        "folder": "Projects/2024",
        "scope": "my-files"
    }
)`}/>

## Notes
- Make sure your Microsoft 365 tenant and app registration are properly configured
- The first authentication will prompt a browser sign-in and create a token file for future use
- File names are case-sensitive
- If multiple files with the same name exist, an error will be raised
- Ensure you have the necessary permissions to access the files
- For more advanced usage, see the [Microsoft Graph API documentation](https://learn.microsoft.com/en-us/graph/overview)

