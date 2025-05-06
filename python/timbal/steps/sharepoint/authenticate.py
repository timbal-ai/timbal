"""
SharePoint Authentication Module

This module provides functionality to authenticate with SharePoint and Microsoft 365,
enabling access to SharePoint sites, files, and other Microsoft 365 resources.

Prerequisites:
-------------
1. Microsoft 365 Setup:
   - Create a Microsoft 365 tenant (if you don't have one)

2. App Registration:
   - Go to Azure Portal (https://portal.azure.com)
   - Navigate to "App registrations"
   - Create a new registration
   - Configure the following:
     - Name: Your app name
     - Supported account types: Single tenant
     - Redirect URI: https://login.microsoftonline.com/common/oauth2/nativeclient (for desktop app)
   - Note down the:
     - Application (client) ID
     - Directory (tenant) ID
     - Create a client secret

3. Configure Permissions:
   - In your app registration, go to "API permissions"
   - Add the following Microsoft Graph permissions:
     - Files.Read.All
     - Sites.Read.All
   - Grant admin consent for these permissions

4. Create Credentials File:
   - Create 'token_sharepoint.json' in this directory
   - Add the following structure:
     {
       "client_id": "your_client_id",
       "client_secret": "your_client_secret",
       "tenant_id": "your_tenant_id"
     }

Note:
-----
- Keep your credentials secure and never commit them to version control
- The token file will be automatically created after first authentication
- Ensure you have the necessary permissions in your app registration
"""

import json
from pathlib import Path

from O365 import Account, FileSystemTokenBackend

# Using SharePoint scope
SCOPES = ['Files.Read.All', 'Sites.Read.All']

def get_token_path() -> Path:
    """Get the path to the token file."""
    script_dir = Path(__file__).parent
    return script_dir / "token_sharepoint.json"

def authenticate() -> tuple[bool, str | None]:
    """Authenticate with SharePoint and save token.
    Returns a tuple of (success, user_identifier)"""
    try:
        token_file = get_token_path()
        
        if not token_file.exists():
            raise Exception("Error: token_sharepoint.json not found")
            
        with open(token_file) as f:
            credentials = json.load(f)
            
        if not all(key in credentials for key in ['client_id', 'client_secret', 'tenant_id']):
            raise Exception("Error: token_sharepoint.json is missing required fields")
            
        token_backend = FileSystemTokenBackend(token_path=token_file.parent, token_filename=token_file.name)
        
        account = Account(
            credentials=(credentials['client_id'], credentials['client_secret']),
            token_backend=token_backend,
            auth_flow_type='authorization',
        )
        
        # Authenticate
        if not account.is_authenticated:
            if account.authenticate(scopes=SCOPES):
                user = account.get_current_user_data()
                user_identifier = user.mail if hasattr(user, 'mail') else user.user_principal_name
                if not user_identifier:
                    raise Exception("Could not determine user identifier")
                return True, user_identifier
            else:
                raise Exception("Failed to authenticate with SharePoint")
                
        # Get user identifier
        user = account.get_current_user_data()
        user_identifier = user.mail if hasattr(user, 'mail') else user.user_principal_name
        if not user_identifier:
            raise Exception("Could not determine user identifier")
            
        return True, user_identifier
        
    except Exception as e:
        raise Exception(f"Error: {e}") from e
    
    

def get_sharepoint_account() -> Account | None:
    """Get authenticated SharePoint account for a specific user."""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        token_file = script_dir / "token_sharepoint.json"
        
        if not token_file.exists():
            raise Exception("Error: Token file not found")
            
        with open(token_file) as f:
            credentials = json.load(f)

        token_backend = FileSystemTokenBackend(token_path=token_file.parent, token_filename=token_file.name)
        
        account = Account(credentials=(credentials['client_id'], credentials['client_secret']), token_backend=token_backend)
        
        if not account.is_authenticated:
            raise Exception("Error: Not authenticated with SharePoint")
            
        return account
        
    except Exception as e:
        raise Exception(f"Error: {e}") from e