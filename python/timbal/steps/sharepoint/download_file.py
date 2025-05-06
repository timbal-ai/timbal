"""
SharePoint File Download Module

This module provides functionality to download files from SharePoint, including
support for both personal and shared files across different scopes.

Features:
---------
- Download files by name or ID
- Search in specific folders
- Support for multiple search scopes:
  - Personal files (my-files)
  - Shared files (shared)
  - All files (all)
- Recursive folder search
- Custom download locations

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
1. Download a file from personal drive:
   >>> download_file("document.docx")

2. Download from a specific folder:
   >>> download_file(
   ...     "report.pdf",
   ...     folder="Projects/2024"
   ... )

3. Download from shared files:
   >>> download_file(
   ...     "presentation.pptx",
   ...     scope="shared"
   ... )

4. Download to specific location:
   >>> download_file(
   ...     "data.xlsx",
   ...     destination="/path/to/save"
   ... )

Note:
-----
- File names are case-sensitive
- If multiple files with the same name exist, an error will be raised
- Ensure you have the necessary permissions to access the files
- The default download location is the current working directory
"""

import os

from O365.drive import Drive, File, Folder

from ...types import Field
from .authenticate import get_sharepoint_account


def search_folder(
    file: str = Field(
        description="The name of the file to find"
    ),
    folder: Folder = Field(
        description="The SharePoint folder"
    ), 
    current_path: str = Field(
        default="",
        description="The current path"
    ),
    items: list[tuple[File, str]] = Field(
        default=[],
        description="The list of items found"
    )
) -> list[tuple[File, str]]:
    """Recursively search for a file in a SharePoint folder.

    This function traverses through a folder and its subfolders to find
    files matching the specified name. It maintains the full path to each found file.

    Args:
        file: Name of the file to search for
        folder: The SharePoint folder to search in
        current_path: Current path in the folder hierarchy (used for recursion)
        items: List to store found files and their paths (used for recursion)

    Returns:
        list[tuple[File, str]]: List of tuples containing:
            - File: The found SharePoint file object
            - str: The full path to the file relative to the search root

    Raises:
        Exception: If there's an error accessing the folder or its contents
    """
    try:
        for item in folder.get_items():
            if isinstance(item, File) and item.name == file:
                full_path = f"{current_path}/{item.name}" if current_path else item.name
                items.append((item, full_path))
            elif isinstance(item, Folder):
                new_path = f"{current_path}/{item.name}" if current_path else item.name
                search_folder(file, item, new_path, items)
    except Exception as e:
        raise Exception(f"Error searching in folder {current_path}: {e}") from e



def find_item(
    drive: Drive = Field(
        description="The SharePoint drive"
    ), 
    file: str = Field(
        description="The name or id of the file to find"
    ), 
    folder: str | None = Field(
        default=None,
        description="The path to the folder to search in"
    ), 
    scope: str = Field(
        default="my-files",
        description="Where to search for the file: 'my-files', 'shared', or 'all'"
    )
) -> File | None:
    """Find a file or folder in SharePoint by name or ID across different scopes.
    The search can be restricted to a specific folder path if provided.

    Args:
        drive: The SharePoint drive instance to search in
        file: Name or ID of the file/folder to find
        folder: Optional path to a specific folder to search in
        scope: Search scope ('my-files', 'shared', or 'all'). Defaults to 'my-files'

    Returns:
        File | None: The found SharePoint file/folder object, or None if not found

    Raises:
        Exception: If the item is not found
        Exception: If multiple items with the same name are found
        Exception: If there's an error accessing the drive or folders
    """
    try:
        items = []
        # Search in regular items recursively
        if scope in ['my-files', 'all']:
            if folder:
                try:
                    folder = drive.get_item_by_path(folder)
                    if not folder:
                        raise Exception(f"Error: Folder '{folder}' not found")
                    search_folder(file, folder, folder, items)
                except Exception:
                    # If folder not found, try searching from root
                    search_folder(file, drive, "", items)
            else:
                # Start from the root
                search_folder(file, drive, "", items)
            
        # Search in shared items recursively
        if scope in ['shared', 'all']:
            shared_items = drive.get_shared_with_me()
            
            for item in shared_items:
                if isinstance(item, File) and item.name == file:
                    items.append((item, item.name))
                elif isinstance(item, Folder):
                    search_folder(file, item, item.name, items)
            
        if not items:
            raise Exception(f"Error: Item '{file}' not found")
            
        if len(items) > 1:
            raise Exception(f"Error: Multiple items found with name '{file}'")
     
        return items[0][0] if isinstance(items[0], tuple) else items[0]
        
    except Exception as e:
        raise Exception(f"Error: {e}") from e



def download_file(
    file: str = Field(
        description="The name or id of the file to find"
    ), 
    destination: str = Field(
        default=None,
        description="The destination directory"
    ),
    folder: str | None = Field(
        default=None,
        description="The SharePoint folder path to search in"
    ),
    scope: str = Field(
        default="my-files",
        description="Where to search for the file: 'my-files', 'shared', or 'all'"
    )
) -> bool:
    """Download a file from SharePoint by searching for it across different scopes.
    This function first locates the file using find_item() and then downloads it
    to the specified destination.
    The search can be restricted to a specific folder path if provided.

    Args:
        file: Name or ID of the file to find and download
        destination: Directory where the file will be downloaded to. If None,
            uses the current working directory
        folder: Optional path to a specific folder to search in
        scope: Search scope ('my-files', 'shared', or 'all'). Defaults to 'my-files'

    Returns:
        bool: True if the file was successfully downloaded
    """
    # Enable calling this step without pydantic model_validate()
    destination = destination.default if hasattr(destination, 'default') else destination
    folder = folder.default if hasattr(folder, 'default') else folder
    scope = scope.default if hasattr(scope, 'default') else scope

    if not destination:
        destination = os.getcwd()
    
    try:
        account = get_sharepoint_account()
        if not account:
            raise Exception("Error: Could not get SharePoint account")
        
        # TODO: Get a specific drive
        drive = account.storage().get_default_drive()
        
        file = find_item(drive, file, folder, scope)
        if not file:
            raise Exception(f"Error: Could not find file {file}")
        
        if not isinstance(file, File):
            raise Exception(f"Error: '{file.name}' is not a file")
        
        if not os.path.isdir(destination):
            raise ValueError(f"Destination '{destination}' is not a valid directory.")
        
        try:
            file.download(destination)
        except Exception as e:
            raise Exception(f"Error: Could not download file {file}: {e}") from e
        
        return True
    
    except Exception as e:
        raise Exception(f"Error: {e}") from e