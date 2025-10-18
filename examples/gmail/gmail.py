"""
The script polls Gmail for new messages and generates drafts for them.

Prerequisites:
- For OAuth: Have 'credentials.json' and 'token.json' files ready
- For Service Account: Have 'credentials.json' and set DELEGATED_USER environment variable
"""

import asyncio
import os
import time
import base64
from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Only needed for OAuth method
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Only needed for Service Account method
from google.oauth2 import service_account

from timbal import Agent
from timbal.state import get_run_context
from dotenv import load_dotenv

load_dotenv()

# Authentication configuration
AUTH_METHOD = os.getenv("AUTH_METHOD", "oauth")  # "oauth" or "service_account"
TOKEN_FILE = os.getenv("TOKEN_FILE")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "credentials.json")
DELEGATED_USER = os.getenv("DELEGATED_USER")  # Required for service account

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.compose'
]


def get_oauth_credentials():
    """Get credentials using OAuth authentication"""
    print("Using OAuth authentication...")
    if not TOKEN_FILE or not os.path.exists(TOKEN_FILE):
        raise Exception("No valid credentials found. Please run setup_credentials.py first.")
    
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            raise Exception("Credentials invalid. Please run setup_credentials.py.")
    
    return creds


def get_service_account_credentials():
    """Get credentials using Service Account with domain-wide delegation"""
    print("Using Service Account authentication...")
    if not DELEGATED_USER:
        raise Exception("DELEGATED_USER environment variable is required for service account authentication")
    
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES)
    
    delegated_creds = credentials.with_subject(DELEGATED_USER)
    print(f"Delegating to user: {DELEGATED_USER}")
    
    return delegated_creds


def get_gmail_credentials():
    """Get Gmail credentials based on AUTH_METHOD"""
    if AUTH_METHOD == "service_account":
        return get_service_account_credentials()
    else:
        return get_oauth_credentials()


class GmailMonitor:
    """Simple Gmail monitor using polling"""
    
    def __init__(self):
        self.gmail_service = None
        self.last_history_id = None

        try:
            self.agent = Agent(
                name="email_response_generator", 
                model="openai/gpt-4o-mini",
                system_prompt="You are an assistant that helps draft professional email responses.",
                post_hook=self.save_draft,
            )  
        except Exception as e:
            raise Exception(f"Error initializing Timbal Agent: {e}")

        
    def initialize_gmail(self):
        """Initialize Gmail API connection"""
        try:
            # Get credentials based on configured auth method
            creds = get_gmail_credentials()
            
            # Build Gmail service
            self.gmail_service = build('gmail', 'v1', credentials=creds)

            # Test connection and get initial history ID
            profile = self.gmail_service.users().getProfile(userId='me').execute()
            print(f"Connected to Gmail: {profile['emailAddress']}")
            
            # Get initial history ID from profile
            self.last_history_id = profile.get('historyId')
            if self.last_history_id:
                print(f"Starting from history ID: {self.last_history_id}")
            else:
                print("No history ID found, will monitor from now")
            
            return True
            
        except Exception as e:
            print(f"Error initializing Gmail API: {e}")
            return False
    

    def get_message_details(self, message_id):
        """Get message details including body content"""
        try:
            message = self.gmail_service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            payload = message['payload']
            headers = payload.get('headers', [])
            
            # Extract headers
            details = {
                'id': message['id'],
                'thread_id': message.get('threadId'),
                'snippet': message.get('snippet', ''),
                'date': None,
                'subject': None,
                'from': None,
                'to': None,
                'body': '',
                'labels': message.get('labelIds', [])
            }
            
            for header in headers:
                name = header['name'].lower()
                if name == 'date':
                    details['date'] = header['value']
                elif name == 'subject':
                    details['subject'] = header['value']
                elif name == 'from':
                    details['from'] = header['value']
                elif name == 'to':
                    details['to'] = header['value']
            
            # Extract message body
            details['body'] = self._extract_message_body(payload)
            
            return details
            
        except HttpError as error:
            print(f"Error getting message details: {error}")
            return None
    

    def _extract_message_body(self, payload):
        """Extract text content from message payload"""
        def extract_text(part):
            if part.get('mimeType') == 'text/plain':
                data = part.get('body', {}).get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8')
            elif part.get('mimeType') == 'text/html':
                data = part.get('body', {}).get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8')
            elif part.get('parts'):
                for subpart in part['parts']:
                    text = extract_text(subpart)
                    if text:
                        return text
            return None
        
        return extract_text(payload) or ''
    

    def is_draft_message(self, message_details):
        """Check if a message is a draft based on its labels"""
        labels = message_details.get('labels', [])
        # Draft messages have the 'DRAFT' label and not 'INBOX'
        return 'DRAFT' in labels and 'INBOX' not in labels
    

    # Post hook
    def save_draft(self):
        """Create a draft response to the original message"""
        span = get_run_context().current_span()
        original_message = span.input['input_email']
        response_text = span.output.content[0].text

        try:
            # Extract sender email from the original message
            from_header = original_message.get('from', '')
            if '<' in from_header and '>' in from_header:
                # Extract email from "Name <email@domain.com>" format
                sender_email = from_header.split('<')[1].split('>')[0].strip()
            else:
                # Assume the entire from header is the email
                sender_email = from_header.strip()
            
            # Create response subject
            original_subject = original_message.get('subject', '')
            if not original_subject.startswith('Re:'):
                response_subject = f"Re: {original_subject}"
            else:
                response_subject = original_subject
            
            # Create draft message
            draft_message = self._create_message(sender_email, response_subject, response_text)
            
            # Create the draft with thread ID to link it to the original conversation
            draft_body = {
                'message': {
                    'raw': draft_message
                }
            }
            
            # Include thread ID if available to link the draft to the original conversation
            if original_message.get('thread_id'):
                draft_body['message']['threadId'] = original_message['thread_id']
            
            draft = self.gmail_service.users().drafts().create(
                userId='me',
                body=draft_body
            ).execute()
            
            print(f"Draft ID: {draft['id']}")
            print(f"To: {sender_email}")
            print(f"Subject: {response_subject}")
            return draft
            
        except Exception as e:
            print(f"Error creating draft: {e}")
            return None
    

    def _create_message(self, to, subject, body):
        """Create a message in Gmail API format"""
        from email.mime.text import MIMEText
        
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        return raw_message


    def _create_email_prompt(self, email):
        # Extract email information
        subject = email.get('subject', 'No Subject')
        sender = email.get('from', 'Unknown Sender')
        body = email.get('body', '')
        snippet = email.get('snippet', '')
        content = body if body else snippet
        
        prompt = f"""
You are an AI assistant that helps draft professional email responses. 

Please write a helpful and appropriate response to this email:

**Received Email:**
From: {sender}
Subject: {subject}
Content: {content}

**Instructions:**
- Write a professional response
- Do not assume anything that is not explicitly stated in the email
- Address any questions or requests appropriately
- Keep the tone friendly but business-appropriate
- If you need more information, ask clarifying questions
- End with a professional closing
- Keep the response concise but complete

**Your Response:**
"""
        return prompt.strip()

    
    async def generate_draft(self, message):
        prompt = self._create_email_prompt(message)
        
        # Generate and create draft
        try:
            response_text = await self.agent(prompt=prompt, input_email=message).collect()
            if response_text:
                print("Generated Response:")
                print("-" * 40)
                print(response_text)
                print("-" * 40)
                
                print("Draft created successfully!")
                print("="*60)
            else:
                print("Failed to generate response")
        except Exception as e:
            print(f"Error generating/creating draft: {e}")
        
        print("="*60)
    
    
    async def check_for_new_messages(self):
        """Check for new messages since last check"""
        try:           
            # Check for new messages since last history ID
            history = self.gmail_service.users().history().list(
                userId='me',
                startHistoryId=self.last_history_id,
                historyTypes=['messageAdded']
            ).execute()
            
            new_messages = []
            for history_record in history.get('history', []):
                for message_added in history_record.get('messagesAdded', []):
                    message_id = message_added['message']['id']
                    message_details = self.get_message_details(message_id)
                    if message_details:
                        # Filter out draft messages
                        if not self.is_draft_message(message_details):
                            new_messages.append(message_details)
                        else:
                            print(f"Skipping draft message: {message_details.get('subject', 'No Subject')}")
            
            if new_messages:
                print(f"\n🔔 Found {len(new_messages)} new message(s) at {datetime.now().strftime('%H:%M:%S')}")
                for message in new_messages:
                    await self.generate_draft(message)
            
            # Update history ID
            if history.get('history'):
                self.last_history_id = history['history'][-1]['id']
            
        except HttpError as error:
            print(f"Error checking for new messages: {error}")

    
    async def start_monitoring(self):
        """Start monitoring Gmail for new messages using polling"""       
        try:
            while True:
                await self.check_for_new_messages()
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            print("Monitor stopped successfully")
    

    async def run(self):
        print("Monitoring Gmail for new messages...")
        print(f"Using polling (checks every 10 seconds)")
        print("=" * 45)
        
        # Initialize Gmail API
        if not self.initialize_gmail():
            return False
        
        # Start monitoring
        await self.start_monitoring()
        return True


async def main():  
    # Create and run monitor
    monitor = GmailMonitor()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
