"""
Google Drive Directory Listing

This module provides functionality to list and explore contents of Google Drive,
including both personal and shared drives. It offers detailed file information
and flexible search options.

Features:
---------
- List files and folders in any drive location
- Search in personal drive, shared drives, or both
- Get detailed file information:
  - File name and ID
  - File size (human-readable format)
  - Owner information
  - Last modified date
  - Sharing status
  - File type (folder/file)
- Save listings to a file
- Filter by scope (personal/shared/all)

Prerequisites:
-------------
1. Google Cloud Project setup:
   - Create a project at https://console.cloud.google.com
   - Enable the Google Drive API:
     https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com

2. Authentication:
   - Configure OAuth 2.0 credentials
   - Save credentials as 'credentials_gdrive.json'
   - For service accounts:
     https://cloud.google.com/iam/docs/service-accounts-create

Usage Examples:
--------------
1. List root directory:
   >>> list_directory()

2. List specific folder:
   >>> list_directory(folder="My Projects")

3. List shared drive contents:
   >>> list_directory(shared_drives="Team Drive")

4. List only shared files:
   >>> list_directory(scope="shared")

5. Save listing to file:
   >>> list_directory(output_path="drive_listing.json")

Note:
-----
- File sizes are displayed in human-readable format (B, KB, MB, GB)
- Dates are shown in local timezone
- Requires appropriate permissions for shared drives
- Maximum of 1000 items per listing
"""

from datetime import datetime
from pathlib import Path
import json
from typing import Any

from googleapiclient.discovery import build

from ...types import Field
from .authenticate import load_api_key, load_credentials


def format_size(
    size: int | None = Field(
        default=None,
        description="The size of the file in bytes"
    )
) -> str:
    """
    Format file size in bytes to human readable format.

    Args:
        size: The size of the file in bytes

    Returns:
        str: The size of the file in human readable format
    """
    if size is None or size == '0':
        return "N/A"
    size = int(size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def item_to_dict(
    item: dict[str, Any] = Field(
        default=None,
        description="The Google Drive item to convert to a dictionary"
    ),
    folder_prefix: str | None = Field(
        default=None,
        description="The folder name to prefix the item path with"
    )
) -> dict[str, Any]:
    """
    Convert a Google Drive item to a dictionary.

    Args:
        item: The Google Drive item to convert to a dictionary
        folder_prefix: The folder name to prefix the item path with

    Returns:
        dict[str, Any]: A dictionary containing information for the item
    """
    is_folder = item['mimeType'] == 'application/vnd.google-apps.folder'
    
    size = item.get('size', '0')
    size_str = format_size(size) if not is_folder else "N/A"
    
    owner = item.get('owners', [{}])[0].get('emailAddress', 'Unknown')
    modified_time = item.get('modifiedTime', 'Unknown')
    if modified_time != 'Unknown':
        modified_time = datetime.fromisoformat(modified_time.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
    
    item_dict = {
        "name": item['name'],
        "type": "folder" if is_folder else "file",
        "id": item['id'],
        "path": f"{folder_prefix}/{item['name']}" if folder_prefix else item['name'],
        "size": size_str,
        "modified": modified_time,
        "owner": owner,
        "shared": item.get('shared', False),
        "multiple_owners": len(item.get('owners', [])) > 1,
        "drive_id": item.get('driveId')
    }
    
    return item_dict



def list_directory(
    folder: str | None = Field(
        default=None,
        description="Optional ID or name of the folder to list"
    ),
    shared_drives: str | None = Field(
        default=None,
        description="Optional ID or name of shared drive to search in"
    ),
    scope: str = Field(
        default="all",
        description="Scope of files to list: 'personal' (personal files), 'shared' (files shared with me), or 'all' (both)"
    ),
    output_path: str | None = Field(
        default=None,
        description="Optional path to save the listing"
    )
) -> list[dict[str, Any]]:
    """List files and folders in Google Drive with detailed information.

    This function provides a comprehensive listing of files and folders in Google Drive,
    including both personal and shared drives. It can filter by scope and save the output
    to a file.

    Args:
        folder: ID or name of the folder to list. If None, lists root directory
        shared_drives: ID or name of shared drive to search in. If None, searches in personal drive
        scope: Scope of files to list:
            - 'personal': Only files owned by you
            - 'shared': Files shared with you or in shared drives
            - 'all': All files (default)
        output_path: Path to save the listing. If None, returns the listing as a list of dictionaries

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing information for each item

    Raises:
        Exception: If there's an error accessing the drive or if no files are found
    """
    # Enable calling this step without pydantic model_validate()
    folder = folder.default if hasattr(folder, 'default') else folder
    shared_drives = shared_drives.default if hasattr(shared_drives, 'default') else shared_drives
    scope = scope.default if hasattr(scope, 'default') else scope
    output_path = output_path.default if hasattr(output_path, 'default') else output_path

    try:
        service = build("drive", "v3", credentials=load_credentials(), developerKey=load_api_key())
        output_path = Path(output_path) if output_path else None
        
        if shared_drives:
            drives = service.drives().list(pageSize=10, fields="drives(id, name)").execute()
            matching_drives = [d for d in drives.get('drives', []) if d['name'].lower() == shared_drives.lower()]
            if matching_drives:
                shared_drives = matching_drives[0]['id']
            else:
                raise Exception(f"Could not find shared drive: {shared_drives}")
        
        if folder:
            query = f"name = '{folder}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            params = {'q': query, 'spaces': 'drive', 'fields': "files(id, name)", 'pageSize': 1}
            if shared_drives:
                params.update({'corpora': 'drive', 'driveId': shared_drives, 'includeItemsFromAllDrives': True, 'supportsAllDrives': True})
            
            results = service.files().list(**params).execute()
            items = results.get('files', [])
            if items:
                folder = items[0]['id']
            else:
                raise Exception(f"Could not find folder: {folder}")
        
        query = "trashed = false"
        if folder:
            query += f" and '{folder}' in parents"
        
        params = {
            'q': query,
            'spaces': 'drive',
            'fields': "files(id, name, mimeType, owners, shared, size, modifiedTime, driveId)",
            'pageSize': 1000,
            'orderBy': 'name',
            'includeItemsFromAllDrives': True,
            'supportsAllDrives': True
        }
        
        if shared_drives:
            params.update({'corpora': 'drive', 'driveId': shared_drives})
        else:
            params['corpora'] = 'allDrives'
        
        results = service.files().list(**params).execute()
        items = results.get('files', [])
        
        if scope == "shared":
            items = [item for item in items if item.get('shared', False) or len(item.get('owners', [])) > 1 or item.get('driveId')]
        elif scope == "personal":
            items = [item for item in items if not item.get('shared', False) and len(item.get('owners', [])) == 1 and not item.get('driveId')]
        
        if not items:
            raise Exception("No files found.")
        
        items_list = [item_to_dict(item, folder) for item in items]
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(items_list, f, indent=2, ensure_ascii=False)
        else:
            return items_list
        
    except Exception as e:
        raise Exception(f"Error listing directory: {e}") from e