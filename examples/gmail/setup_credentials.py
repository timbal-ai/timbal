#!/usr/bin/env python3
"""
Gmail API Credentials Setup Script

This script sets up OAuth2 credentials for the Gmail API.
You'll need to create a Google Cloud Project and enable the Gmail API first.
"""

import os
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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
    
    # Check if credentials file exists
    if not os.path.exists(credentials_file):
        print(f"{credentials_file} not found!")
        return None
    
    # Check if token file exists (user has already authenticated)
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If there are no valid credentials, let the user log in
    if not creds or not creds.valid:
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
        # Get user profile to test connection
        profile = service.users().getProfile(userId='me').execute()
        print(f"Successfully connected to Gmail")
        print(f"Email: {profile.get('emailAddress')}")
        return service
    except Exception as e:
        print(f"Failed to connect to Gmail API: {e}")
        return None

if __name__ == "__main__":
    print("Gmail API Credentials Setup")
    print("=" * 40)
    
    credentials = setup_credentials()

    if credentials:
        service = test_connection(credentials)

        if service:
            print("\n Setup complete!")
        else:
            print("\nSetup failed. Please check your credentials and try again.")
    else:
        print("\nPlease complete the credential setup first.")
