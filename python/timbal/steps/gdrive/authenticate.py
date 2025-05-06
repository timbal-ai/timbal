"""
Google Drive API Authentication

This module handles authentication with the Google Drive API, providing access to:
- Personal Google Drive files and folders
- Shared drives and team drives
- Google Workspace files (Docs, Sheets, Slides, etc.)

Authentication Setup:
-------------------
1. Create a Google Cloud Project:
   - Go to Google Cloud Console (https://console.cloud.google.com)
   - Create a new project or select an existing one
   - Enable the Google Drive API:
     https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com

2. Configure OAuth 2.0:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application" as the application type
   - Download the credentials and save as 'credentials_gdrive.json' in this directory

3. First-time Authentication:
   - Run this script
   - A browser window will open asking you to sign in to your Google account
   - Grant the requested permissions
   - The script will create a 'token.json' file for future authentication

4. Service Account (Optional):
   - For server-side applications, create a service account:
     https://cloud.google.com/iam/docs/service-accounts-create
   - Download the service account key and save it securely

Note: The 'token.json' file will be automatically created and used for subsequent runs.
Keep your credentials secure and never commit them to version control.
"""

import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Using full drive scope to access shared files
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly"
]

# Mapping of Google Workspace MIME types to export formats
GOOGLE_MIME_TYPES = {
    'application/vnd.google-apps.document': {
        'mime': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'ext': '.docx'
    },
    'application/vnd.google-apps.spreadsheet': {
        'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'ext': '.xlsx'
    },
    'application/vnd.google-apps.presentation': {
        'mime': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'ext': '.pptx'
    },
    'application/vnd.google-apps.drawing': {
        'mime': 'image/png',
        'ext': '.png'
    }
}

def load_api_key():
    """Load API key from credentials file."""
    script_dir = Path(__file__).parent
    credentials_file = script_dir / "credentials_gdrive.json"
    
    try:
        with open(credentials_file) as f:
            creds_data = json.load(f)
            return creds_data.get("api_key")
    except Exception as e:
        raise Exception(f"Error loading API key: {e}") from e


def load_credentials():
    """Load credentials from various sources."""
    script_dir = Path(__file__).parent
    credentials_file = script_dir / "credentials_gdrive.json"
    token_file = script_dir / "token.json"
    
    creds = None
    
    # From Oauth Client
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_file.exists():
                raise Exception(f"Error: Credentials file not found at {credentials_file}")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), SCOPES
            )
            creds = flow.run_local_server(port=0)
        if creds:
            with open(token_file, "w") as token:
                token.write(creds.to_json())

    return creds