---
title: Twilio
sidebar: 'docsSidebar'
---

import CodeBlock from '@site/src/theme/CodeBlock';

# Twilio Integration

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Send WhatsApp messages programmatically with Twilio integration for both free-form and template-based messaging.
</h2>

---

Timbal integrates with Twilio to enable sending WhatsApp messages programmatically. 

This integration allows you to send both free-form and template-based WhatsApp messages directly from your Timbal workflows.

## Prerequisites

Before using the Twilio integration, you'll need:

1. A Twilio account – [Sign up here](https://www.twilio.com/try-twilio)
2. A WhatsApp-enabled Twilio phone number
3. Your Twilio Account SID, Auth Token, and WhatsApp From number
4. Store your credentials in environment variables: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`


## <span style={{color: 'var(--timbal-purple)'}}><strong>Send WhatsApp Message</strong></span>

### Description
The **send_whatsapp_message** step allows you to send a free-form WhatsApp message to a user.

### Example
<CodeBlock language="python" code ={`from timbal.steps.twilio.whatsapp import send_whatsapp_message

await send_whatsapp_message(
    to="+1234567890",
    message="Hello from Timbal!"
)`}/>

### Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `to` | `str` | The WhatsApp account to send the message to in E.164 format | Yes |
| `message` | `str` | The message to send (max 1600 characters; longer messages will be split) | Yes |

## <span style={{color: 'var(--timbal-purple)'}}><strong>Send WhatsApp Template</strong></span>

### Description
The **send_whatsapp_template** step allows you to send a pre-approved WhatsApp template message to a user, with parameters.

### Example
<CodeBlock language="python" code ={`from timbal.steps.twilio.whatsapp import send_whatsapp_template

await send_whatsapp_template(
    to="+1234567890",
    template_sid="your-template-sid",
    template_params={"name": "John"}
)`}/>

### Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `to` | `str` | The WhatsApp account to send the message to in E.164 format | Yes |
| `template_sid` | `str` | The template SID to send the message with | Yes |
| `template_params` | `dict` | Parameters to fill in the template (must match template definition) | Yes |

## Agent Integration Example

<CodeBlock language="python" code ={`from timbal.steps.twilio.whatsapp import send_whatsapp_message
from timbal import Agent

agent = Agent(
    tools=[send_whatsapp_message]
)

response = await agent.complete(
                prompt={
                    "to": "+1234567890", 
                    "message": "Hello from Agent!"
                }
        )`}/>

## Notes
- For more advanced usage, see the [Twilio WhatsApp API documentation](https://www.twilio.com/docs/whatsapp/api).