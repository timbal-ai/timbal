"""
Gmail API Authentication Setup Guide

This module handles authentication with the Gmail API. Follow these steps to set up authentication:

1. Prerequisites:
   - A Google account with Gmail access

2. Create Google Cloud Project and Enable Gmail API:
   - Go to Google Cloud Console (https://console.cloud.google.com)
   - Create a new project or select an existing one
   - Enable the Gmail API for your project
   - Configure the OAuth consent screen

3. Create OAuth 2.0 Credentials:
   - In Google Cloud Console, go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application" as the application type
   - Download the credentials and save as 'gmail_credentials.json' in this directory

4. First-time Authentication:
   - Run this script
   - A browser window will open asking you to sign in to your Google account
   - Grant the requested permissions
   - The script will create a 'token.json' file for future authentication

Note: The 'token.json' file will be automatically created and used for subsequent runs.
Keep your credentials secure and never commit them to version control.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

TOKEN_PATH = Path(__file__).parent / 'token.json'
CREDENTIALS_PATH = Path(__file__).parent / 'gmail_credentials.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_client():
    """Create a Gmail client with cached credentials."""
    # Try to load cached credentials
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    
    return build("gmail", "v1", credentials=creds)