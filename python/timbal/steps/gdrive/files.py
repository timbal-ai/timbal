"""
Google Drive File Download Module

This module provides functionality to download files from Google Drive,
including support for both personal and shared drives, as well as Google Workspace files.

Setup:
------
You need the OAuth credentials and the API key.
First, you need to authenticate with Google Drive. Run the authenticate.py script to get the token.json file.

Features:
---------
- Download files by name or ID
- Support for shared drives and files
- Automatic Google Workspace file conversion (Docs, Sheets, Slides)
- Search files in specific folders
- Custom output paths

Supported File Types:
-------------------
- Regular files (PDF, images, documents, etc.)
- Google Workspace files: Google Docs, Google Sheets, Google Slides, Google Drawings

Usage Examples:
--------------
1. Download a file by name:
   >>> download_file("report.docx")

2. Download a file by ID:
   >>> download_file("1abc123xyz")

3. Download from a specific folder:
   >>> download_file(
   ...     "report.docx",
   ...     folder_name="Project Files"
   ... )

4. Download from a shared drive:
   >>> download_file(
   ...     "report.docx",
   ...     folder_name="Team Drive",
   ...     use_shared_drive=True
   ... )

5. Download to a specific location:
   >>> download_file(
   ...     "report.docx",
   ...     output_path="/path/to/save"
   ... )

Note:
-----
- Google Workspace files are automatically converted to compatible formats
- The original file name is preserved unless specified otherwise
"""

import io
from pathlib import Path

from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload

from ...types import Field
from .authenticate import GOOGLE_MIME_TYPES, load_api_key, load_credentials


def find_folder_by_name(
    service: Resource = Field(
        description="Google Drive service instance"
    ), 
    folder_name: str = Field(
        description="Name of the folder to find"
    ), 
    shared_drive: str | None = Field(
        default=None,
        description="Optional name of the shared drive to search in"
    )
) -> str | None:
    """
    Find a folder by its name.
    
    Args:
        service: Google Drive service instance
        folder_name: Name of the folder to find
        shared_drive: Optional name of the shared drive to search in
        
    Returns:
        Folder ID if found, None otherwise
    """
    # Enable calling this step without pydantic model_validate()
    shared_drive = shared_drive.default if hasattr(shared_drive, 'default') else shared_drive

    try:
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        
        params = {
            'q': query,
            'spaces': 'drive',
            'fields': "files(id, name, parents)",
            'pageSize': 10,
        }
        
        if shared_drive:
            params.update({
                'corpora': 'drive',
                'driveId': shared_drive,
                'includeItemsFromAllDrives': True,
                'supportsAllDrives': True
            })
        
        results = service.files().list(**params).execute()
        
        folders = results.get('files', [])
        
        if not folders:
            raise Exception(f"No folders found with name: {folder_name}")
            
        if len(folders) > 1:
            raise Exception(f"Found {len(folders)} folders named '{folder_name}'. Please specify the folder ID or drive to narrow down the search.")
            
        return folders[0]['id']
            
    except Exception as e:
        raise Exception(f"Error finding folder: {e}") from e



def get_all_subfolder_ids(
    service: Resource = Field(
        description="Google Drive service instance"
    ), 
    parent_id: str = Field(
        description="ID of the parent folder"
    ), 
    shared_drive: str | None = Field(
        default=None,
        description="Optional name of the shared drive to search in"
    )
) -> list[str]:
    """Recursively retrieve all subfolder IDs from a Google Drive folder.

    This function traverses the folder hierarchy starting from the specified parent folder,
    collecting IDs of all subfolders at any depth level.

    Args:
        service: Authenticated Google Drive API service instance
        parent_id: The ID of the parent folder to start the search from
        folder_ids: Optional list to store the found folder IDs. If None, a new list is created

    Returns:
        list[str]: A list of folder IDs, including the parent folder and all its subfolders
    """
    folder_ids = [parent_id]
    query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    
    search_params = {
        'q': query,
        'spaces': 'drive',
        'fields': 'files(id)',
        'pageSize': 1000,
    }
    
    if shared_drive:
        search_params.update({
            'corpora': 'drive',
            'driveId': shared_drive,
            'includeItemsFromAllDrives': True,
            'supportsAllDrives': True
        })

    results = service.files().list(**search_params).execute()
    subfolders = results.get('files', [])
    
    for folder in subfolders:
        folder_ids.extend(get_all_subfolder_ids(service, folder['id'], shared_drive))
    
    return folder_ids



def search_file_by_name(
    service: Resource = Field(
        description="Google Drive service instance"
    ), 
    name: str = Field(
        description="Name of the file to search for"
    ), 
    folder: str | None = Field(
        default=None,
        description="Optional path or ID of the folder to search in"
    ), 
    scope: str = Field(
        default="all",
        description="Search scope: 'all' for all drives, 'shared-with-me' for shared drives only, 'my-drive' for personal drive only"
    ),
    recursive: bool = Field(
        default=False,
        description="Whether to search recursively in subfolders"
    ),
    shared_drive: str | None = Field(
        default=None,
        description="Name of the shared drive (for root recursion)"
    )
) -> dict | None:
    """
    Search for a file by name in Google Drive.
    
    Args:
        service: Google Drive service instance
        name: Name of the file to search for
        folder_path: Optional path or ID of the folder to search in
        scope: Search scope - 'all' for all drives, 'shared-with-me' for shared drives only, 'my-drive' for personal drive only
        recursive: Whether to search recursively in subfolders
        shared_drive: Name of the shared drive (for root recursion)
        
    Returns:
        File metadata if found, None otherwise
    """
    try:
        # Check if the name is actually a file ID
        try:
            return service.files().get(
                fileId=name,
                fields='id, name, mimeType, owners',
                supportsAllDrives=True
            ).execute()
        except Exception:
            # If it's not a file ID, continue with name search
            pass

        search_name = name.strip()
        query_parts = [
            "trashed = false",
            f"name contains '{search_name}'"
        ]
        
        search_params = {
            'q': " and ".join(query_parts),
            'spaces': 'drive',
            'fields': "files(id, name, mimeType, owners, shared, driveId)",
            'pageSize': 100,
            'orderBy': 'name'
        }
        
        if folder:
            query_parts.append(f"'{folder}' in parents")
            search_params.update({
                'corpora': 'allDrives',
                'includeItemsFromAllDrives': True,
                'supportsAllDrives': True
            })
        else:
            if shared_drive:
                search_params.update({
                    'corpora': 'drive',
                    'driveId': shared_drive,
                    'includeItemsFromAllDrives': True,
                    'supportsAllDrives': True
                })
            elif scope == "shared-with-me":
                search_params.update({
                    'corpora': 'allDrives',
                    'includeItemsFromAllDrives': True,
                    'supportsAllDrives': True
                })
            elif scope == "my-drive":
                search_params.update({
                    'corpora': 'user',
                    'includeItemsFromAllDrives': False
                })
            else:
                search_params.update({
                    'corpora': 'allDrives',
                    'includeItemsFromAllDrives': True,
                    'supportsAllDrives': True
                })
        
        if recursive and shared_drive:
            search_params.update({
                'q': f"name contains '{search_name}' and trashed = false",
                'corpora': 'drive',
                'driveId': shared_drive,
                'includeItemsFromAllDrives': True,
                'supportsAllDrives': True
            })
        
        results = service.files().list(**search_params).execute()
        items = results.get('files', [])
        
        if not shared_drive:
            if scope == "shared-with-me":
                items = [item for item in items if item.get('shared', False) or len(item.get('owners', [])) > 1 or item.get('driveId')]
            elif scope == "my-drive":
                items = [item for item in items if not item.get('shared', False) and len(item.get('owners', [])) == 1 and not item.get('driveId')]
        
        if not items:
            return None
        
        if len(items) == 1:
            return items[0]
        
        if folder:
            folder_id = None
            try:
                folder_metadata = service.files().get(
                    fileId=folder,
                    fields='id, name, mimeType',
                    supportsAllDrives=True
                ).execute()
                if folder_metadata['mimeType'] == 'application/vnd.google-apps.folder':
                    folder_id = folder_metadata['id']
                else:
                    raise Exception(f"Specified ID is not a folder: {folder}")
            except Exception:
                # If not a folder ID, try to find by name
                folder_id = find_folder_by_name(service, folder, shared_drive)
            
            if folder_id:
                query_parts = [
                    "trashed = false",
                    f"name contains '{search_name}'",
                    f"'{folder_id}' in parents"
                ]
                search_params['q'] = " and ".join(query_parts)
                results = service.files().list(**search_params).execute()
                items = results.get('files', [])
                
                if scope == "shared-with-me":
                    items = [item for item in items if item.get('shared', False) or len(item.get('owners', [])) > 1 or item.get('driveId')]
                elif scope == "my-drive":
                    items = [item for item in items if not item.get('shared', False) and len(item.get('owners', [])) == 1 and not item.get('driveId')]
                
                if len(items) == 1:
                    return items[0]
        
        if len(items) > 1:
            raise Exception(f"Found {len(items)} files named '{search_name}'. Please specify the file ID or folder name to narrow down the search.")
        
        return items[0]
        
    except Exception as e:
        raise Exception(f"Error searching for file: {e}") from e



def download_gdrive_file(
    file_id: str = Field(
        description="The ID of the file to download"
    ), 
    output_path: Path | None = Field(
        default=None,
        description="Where to save the file. If None, saves in current directory with original name"
    ), 
    is_shared_drive: bool = Field(
        default=False,
        description="Whether the file is in a shared drive"
    )
) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: The ID of the file to download
        output_path: Where to save the file. If None, saves in current directory with original name
        is_shared_drive: Whether the file is in a shared drive
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        credentials = load_credentials()
        if not credentials:
            return False

        api_key = load_api_key()
        if not api_key:
            raise Exception("Error: API key not found in credentials file")
            
        service = build("drive", "v3", credentials=credentials, developerKey=api_key)
        
        file_params = {
            'fileId': file_id,
            'fields': "name, mimeType, shared"
        }
        if is_shared_drive:
            file_params['supportsAllDrives'] = True
        
        file_metadata = service.files().get(**file_params).execute()
        
        if output_path is None:
            output_path = Path(file_metadata["name"])
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / file_metadata["name"]
        
        mime_type = file_metadata.get('mimeType')
        # Google Workspace file that needs to be exported
        if mime_type in GOOGLE_MIME_TYPES:
            export_details = GOOGLE_MIME_TYPES[mime_type]
            output_path = output_path.with_suffix(export_details['ext'])
            request = service.files().export_media(
                fileId=file_id,
                mimeType=export_details['mime']
            )
        else:
            download_params = {'fileId': file_id}
            if is_shared_drive:
                download_params['supportsAllDrives'] = True
            request = service.files().get_media(**download_params)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        while not downloader.next_chunk()[1]:
            pass
        
        with open(output_path, "wb") as file:
            file.write(file_buffer.getvalue())
        
        return True
        
    except Exception as e:
        raise Exception(f"Error downloading file: {e}") from e



def download_file(
    file: str = Field(
        description="Name or ID of the file to download"
    ), 
    folder: str | None = Field(
        default=None,
        description="Optional name or ID of the folder to search in"
    ), 
    output_path: str | None = Field(
        default=None,
        description="Optional path where to save the file (defaults to current directory)"
    ), 
    scope: str = Field(
        default="all",
        description="Search scope: 'all' for all drives, 'shared-with-me' for shared drives only, 'my-drive' for personal drive only"
    ),
    shared_drive: str | None = Field(
        default=None,
        description="Optional name of the shared drive to search in"
    ),
    recursive: bool = Field(
        default=False,
        description="Whether to search recursively in subfolders"
    )
) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        file_name_or_id: Name or ID of the file to download
        folder_name: Optional name or ID of the folder to search in
        output_path: Optional path where to save the file (defaults to current directory)
        scope: Search scope - 'all' for all drives, 'shared-with-me' for shared drives only, 'my-drive' for personal drive only
        shared_drive: Optional name of the shared drive to search in
        recursive: Whether to search recursively in subfolders
        
    Returns:
        True if download was successful, False otherwise
    """
    # Enable calling this step without pydantic model_validate()
    folder = folder.default if hasattr(folder, 'default') else folder
    output_path = output_path.default if hasattr(output_path, 'default') else output_path
    scope = scope.default if hasattr(scope, 'default') else scope
    shared_drive = shared_drive.default if hasattr(shared_drive, 'default') else shared_drive
    recursive = recursive.default if hasattr(recursive, 'default') else recursive

    try:
        credentials = load_credentials()
        if not credentials:
            raise Exception("Error: Could not load credentials. Please run authenticate.py first.")
            
        api_key = load_api_key()
        if not api_key:
            raise Exception("Error: API key not found in credentials file")
            
        service = build("drive", "v3", credentials=credentials, developerKey=api_key)
        output_path = Path(output_path) if output_path else None
        
        # First try to use the name as a file ID
        try:
            file_metadata = service.files().get(
                fileId=file,
                fields='id, name, mimeType',
                supportsAllDrives=True
            ).execute()
            return download_gdrive_file(
                file_metadata['id'],
                output_path,
                file_metadata['mimeType'].startswith('application/vnd.google-apps')
            )
        except Exception:
            pass
        
        actual_drive_id = None
        if shared_drive:
            try:
                # Try as drive ID
                drive = service.drives().get(driveId=shared_drive).execute()
                actual_drive_id = drive['id']
            except Exception:
                # Try to find by drive name
                try:
                    drives = service.drives().list(
                        pageSize=10,
                        fields="drives(id, name)"
                    ).execute()
                    
                    matching_drives = [d for d in drives.get('drives', []) if d['name'].lower() == shared_drive.lower()]
                    if matching_drives:
                        actual_drive_id = matching_drives[0]['id']
                    else:
                        raise Exception(f"Could not find shared drive with name or ID: {shared_drive}")
                except Exception as e:
                    raise Exception(f"Error searching for shared drive: {e}") from e
        
        folder_id = None
        if folder:
            try:
                # Try as folder ID
                folder_metadata = service.files().get(
                    fileId=folder,
                    fields='id, name, mimeType',
                    supportsAllDrives=True
                ).execute()
                if folder_metadata['mimeType'] == 'application/vnd.google-apps.folder':
                    folder_id = folder_metadata['id']
            except Exception as e:
                # Search by folder name
                folder_id = find_folder_by_name(service, folder, actual_drive_id)
                if not folder_id:
                    raise Exception(f"Could not find folder: {folder}") from e
        
        file_metadata = search_file_by_name(
            service, 
            file, 
            folder_id, 
            scope=scope,
            recursive=recursive,
            shared_drive=actual_drive_id
        )
        
        if not file_metadata:
            raise Exception(f"File not found: {file}")
            
        return download_gdrive_file(
            file_metadata['id'], 
            output_path, 
            file_metadata['mimeType'].startswith('application/vnd.google-apps')
        )
        
    except Exception as e:
        raise Exception(f"Error downloading file: {e}") from e