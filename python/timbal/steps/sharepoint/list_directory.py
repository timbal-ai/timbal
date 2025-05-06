"""
SharePoint Directory Listing Module

This module provides functionality to list and explore files and folders in SharePoint,
with support for both personal and shared files across different scopes.

Features:
---------
- List files and folders in any SharePoint location
- Support for multiple search scopes:
  - Personal files (my-files)
  - Shared files (shared)
  - All files (all)
- Detailed file/folder information:
  - Name and type
  - Size (human-readable format)
  - Creation and modification dates
  - Owner information
  - Web URL
- Optional JSON output to file
- Sorting by name
- Path-based navigation

Prerequisites:
-------------
1. Microsoft 365 Setup:
   - Create a Microsoft 365 tenant (if you don't have one)

2. App Registration:
   - Go to Azure Portal (https://portal.azure.com)
   - Navigate to "App registrations"
   - Create a new registration with:
     - Name: Your app name
     - Supported account types: Single tenant
     - Redirect URI: https://login.microsoftonline.com/common/oauth2/nativeclient
   - Configure required permissions:
     - Files.Read.All
     - Sites.Read.All

3. Authentication:
   - Run authenticate.py to get the token_sharepoint.json file
   - Ensure the token file contains valid credentials

Usage Examples:
--------------
1. List all files in personal drive:
   >>> list_directory(scope="my-files")

2. List contents of a specific folder:
   >>> list_directory(
   ...     folder="Projects/2024",
   ...     scope="my-files"
   ... )

3. List shared files:
   >>> list_directory(scope="shared")

4. Save output to JSON file:
   >>> list_directory(
   ...     scope="all",
   ...     output_path="sharepoint_contents.json"
   ... )

Note:
-----
- File/folder names are case-sensitive
- The output includes detailed metadata for each item
- Items are sorted alphabetically by name
- Empty folders are included in the listing
- Web URLs are provided for direct access to items
"""

import json
from typing import Any

from O365.drive import File, Folder

from ...types import Field
from .authenticate import get_sharepoint_account


def format_size(size: int | None) -> str:
    """Format file size in bytes to human readable format."""
    if size is None:
        return "N/A"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"



def item_to_dict(
    item: Any = Field(
        default=None,
        description="The item to convert to a dictionary"
    ),
    folder_prefix: str | None = Field(
        default=None,
        description="The folder name to prefix the item path with"
    )
) -> dict[str, Any]:
    """Convert a SharePoint item to a dictionary."""
    if isinstance(item, Folder):
        item_type = "folder"
    elif isinstance(item, File):
        item_type = "file"
    else:
        item_type = "unknown"
    item_dict = {
        "name": item.name,
        "type": item_type,
        "path": f"{folder_prefix}/{item.name}" if folder_prefix else item.name
    }
    item_dict.update({
        "size": format_size(item.size) if item_type == "file" and hasattr(item, 'size') else "N/A",
        "modified": item.modified.strftime('%Y-%m-%d %H:%M:%S') if hasattr(item, 'modified') and item.modified else "N/A",
        "created": item.created.strftime('%Y-%m-%d %H:%M:%S') if hasattr(item, 'created') and item.created else "N/A",
        "owner": item.owner.user.display_name if hasattr(item, 'owner') and item.owner else "N/A",
        "web_url": item.web_url if hasattr(item, 'web_url') else "N/A"
    })
    return item_dict



def list_directory(
    folder: str | None = Field(
        default=None,
        description="The SharePoint folder path"
    ),
    scope: str = Field(
        default="my-files",
        description="Where to search for files: 'my-files', 'shared', or 'all'"
    ),
    output_path: str | None = Field(
        default=None,
        description="The path to save the output"
    )
) -> list[dict[str, Any]]:
    """List files and folders in SharePoint.
    
    Args:
        drive: The SharePoint drive to use
        folder: The folder path to list contents from
        scope: Where to search for files ('my-files', 'shared', or 'all')
        output_path: Path to save the output JSON file
        
    Returns:
        List of dictionaries containing file/folder information
        
    Raises:
        Exception: If there's an error accessing SharePoint or processing files
    """
    # Extract values from Field objects
    folder = folder.default if hasattr(folder, 'default') else folder
    scope = scope.default if hasattr(scope, 'default') else scope
    output_path = output_path.default if hasattr(output_path, 'default') else output_path

    account = get_sharepoint_account()
    if not account:
        raise Exception("Error: Could not get SharePoint account")
    
    # TODO: Get a specific drive
    drive = account.storage().get_default_drive()
    
    try:
        items = []

        if scope in ["my-files", "all"]:
            if folder:
                folder_item = drive.get_item_by_path(folder)
                if folder_item:
                    my_items = list(folder_item.get_items())
                    my_items.sort(key=lambda x: x.name)
                    items.extend([item_to_dict(item, folder) for item in my_items])
            else:
                my_items = list(drive.get_items())
                my_items.sort(key=lambda x: x.name)
                items.extend([item_to_dict(item, "") for item in my_items])

        if scope in ["shared", "all"]:
            shared_items = list(drive.get_shared_with_me())
            if folder:
                shared_folder = None
                for item in shared_items:
                    if isinstance(item, Folder) and item.name == folder:
                        shared_folder = item
                        break
                if shared_folder:
                    shared_inside = list(shared_folder.get_items())
                    shared_inside.sort(key=lambda x: x.name)
                    items.extend([item_to_dict(item, folder) for item in shared_inside])
            else:
                shared_root_items = [item for item in shared_items if not hasattr(item, 'parent_reference') or not getattr(item.parent_reference, 'path', None)]
                shared_root_items.sort(key=lambda x: x.name)
                items.extend([item_to_dict(item, "") for item in shared_root_items])

        if output_path:
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
        else:
            return items

    except Exception as e:
        raise Exception(f"Error listing directory: {e}") from e