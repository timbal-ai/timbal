# Gmail Monitor with Timbal

This example demonstrates how to monitor Gmail for new messages and automatically generate draft responses using Timbal AI agents.

## Required Packages

```bash
google-api-python-client
googleapiclient
google-auth-oauthlib
```

## Authentication Methods

The script supports two authentication methods:

### Method 1: OAuth 2.0 (Personal/Desktop Apps)

**Use this if:**
- You're using a personal Gmail account
- You want users to authenticate with their own Google account

**Setup:**
1. Create OAuth credentials in Google Cloud Console
2. Download `credentials.json`
3. Run `python setup_credentials.py` to generate `token.json`
4. Set environment variables:
```bash
AUTH_METHOD=oauth
TOKEN_FILE=token.json
CREDENTIALS_FILE=credentials.json
```

### Method 2: Service Account (Google Workspace)

**Use this if:**
- You have a Google Workspace account
- You need domain-wide delegation

**Setup:**
1. Create a service account in Google Cloud Console
2. Download service account key as `credentials.json`
3. Configure domain-wide delegation in Google Workspace Admin
4. Set environment variables:
```bash
AUTH_METHOD=service_account
CREDENTIALS_FILE=credentials.json
DELEGATED_USER=your-email@yourdomain.com
```

## Environment Variables

Create a `.env` file with the following variables:

```bash
# Authentication method: "oauth" or "service_account"
AUTH_METHOD=oauth

CREDENTIALS_FILE=credentials.json
OPENAI_API_KEY=your-api-key

# For OAuth
TOKEN_FILE=token.json

# For Service Account
DELEGATED_USER=your-email@yourdomain.com
```

## Running the Script

```bash
python gmail.py
```

The script will:
1. Connect to Gmail using your chosen authentication method
2. Monitor for new incoming emails
3. Generate draft responses using Timbal AI
4. Save drafts to the email thread

