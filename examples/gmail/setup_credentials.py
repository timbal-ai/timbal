#!/usr/bin/env python3
"""
Gmail API Credentials Setup Script (OAuth Method Only)

This script sets up OAuth2 credentials for the Gmail API.
You'll need to create a Google Cloud Project and enable the Gmail API first.

Note: This script is only for OAuth authentication. Service Account
authentication doesn't require this setup script.
"""

import os
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

TOKEN_FILE = os.getenv("TOKEN_FILE")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE")

# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.compose'
]

def setup_credentials():
    """Set up OAuth2 credentials for Gmail API"""
    
    creds = None
    token_file = TOKEN_FILE
    credentials_file = CREDENTIALS_FILE
    
    if not os.path.exists(credentials_file):
        print(f"{credentials_file} not found!")
        return None
    
    # If user has already authenticated, load the credentials
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If there are no valid credentials, let the user log in
    if not creds or not creds.valid:
        if os.path.exists(token_file):
            os.remove(token_file)
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print("Starting OAuth2 authentication flow...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        print("Credentials saved successfully!")
    
    return creds

def test_connection(creds):
    """Test the Gmail API connection"""
    try:
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        print(f"✓ Connected to: {profile.get('emailAddress')}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Gmail OAuth Setup")
    print("=" * 50)
    
    credentials = setup_credentials()

    if credentials:
        print("\nTesting connection...")
        if test_connection(credentials):
            print("\n✓ Setup complete! You can now run gmail.py")
        else:
            print("\n✗ Setup failed")
    else:
        print("\n✗ Failed to create credentials")
