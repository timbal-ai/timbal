---
title: Human In-the-Loop Workflow
sidebar: 'examples'
---

import React from 'react';
import Link from '@docusaurus/Link';
import styles from '../../src/css/examples.module.css';
import CodeBlock from '@site/src/theme/CodeBlock';

Use human-in-the-loop workflows to pause execution at specific steps for human input, decision-making, or tasks that require judgment beyond automation.

## Suspend workflow

In this example, the workflow pauses until user input is received. Execution is suspended at a specific step and only resumes once the required confirmation is provided.

<CodeBlock language="python" code={`from timbal.core import Workflow, Tool
from typing import Optional
import asyncio

def step1(value: int) -> dict:
    """Passes value from input to output."""
    return {"value": value}

def step2(value: int, confirm: Optional[bool] = None) -> dict:
    """Pauses until user confirms, then processes the confirmation."""
    if confirm is None:
        # This step would typically suspend execution and wait for human input
        # In a real implementation, you might use a webhook, API endpoint, or UI interaction
        print(f"Workflow paused at step2. Current value: {value}")
        print("Waiting for human confirmation...")
        
        # Simulate human input (in practice, this would come from external source)
        # For demonstration, we'll use a simple input() call
        try:
            user_input = input(f"Confirm processing value {value}? (y/n): ").lower().strip()
            confirm = user_input in ['y', 'yes', 'true']
        except (EOFError, KeyboardInterrupt):
            confirm = False
        
        print(f"Human input received: {confirm}")
    
    return {
        "value": value,
        "confirmed": confirm
    }

# Create the human-in-the-loop workflow
human_in_loop_workflow = (
    Workflow(name="human-in-loop-workflow")
    .step(step1)
    .step(step2)
    .link("step1", "step2")
)

# Alternative: Using Tool class for more complex human interaction
def human_confirmation_tool(value: int, confirm: Optional[bool] = None) -> dict:
    """Tool that handles human confirmation for workflow steps."""
    if confirm is None:
        print(f"=== HUMAN REVIEW REQUIRED ===")
        print(f"Value to process: {value}")
        print(f"Please review and confirm this value.")
        
        # In a real implementation, this would trigger a human review process
        # For now, we'll simulate it with user input
        try:
            user_input = input("Confirm processing? (y/n): ").lower().strip()
            confirm = user_input in ['y', 'yes', 'true']
        except (EOFError, KeyboardInterrupt):
            confirm = False
        
        print(f"Human decision: {confirm}")
    
    return {
        "value": value,
        "confirmed": confirm,
        "human_reviewed": True
    }

human_confirmation_tool_instance = Tool(
    name="human_confirmation",
    description="Pauses workflow for human confirmation",
    handler=human_confirmation_tool
)

tool_human_workflow = (
    Workflow(name="tool-human-workflow")
    .step(step1)
    .step(human_confirmation_tool_instance)
    .link("step1", "human_confirmation")
)
`}/>

## How human-in-the-loop works in Timbal

1. **Workflow Suspension**: Steps can pause execution and wait for external input
2. **Human Input**: External systems (UI, API, webhook) provide the required input
3. **Resume Execution**: Workflow continues once input is received
4. **Flexible Integration**: Can integrate with any human interaction system
5. **Error Handling**: Graceful handling of timeouts and user cancellations

## Example usage

<CodeBlock language="python" code={`import asyncio

async def main():
    print("=== Testing Human-in-the-Loop Workflow ===")
    
    # Test with value 42
    print("Starting workflow with value: 42")
    result = await human_in_loop_workflow(value=42).collect()
    
    print(f"Workflow completed!")
    print(f"Final value: {result.output['value']}")
    print(f"Confirmed: {result.output.get('confirmed', 'N/A')}")
    
    # Test with tool version
    print("\n=== Testing Tool-based Human Workflow ===")
    result2 = await tool_human_workflow(value=100).collect()
    
    print(f"Tool workflow completed!")
    print(f"Final value: {result2.output['value']}")
    print(f"Confirmed: {result2.output.get('confirmed', 'N/A')}")
    print(f"Human reviewed: {result2.output.get('human_reviewed', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
`}/>

## Advanced human-in-the-loop patterns

<CodeBlock language="python" code={`# Human approval with multiple options
def human_approval_step(value: int, approval_type: Optional[str] = None) -> dict:
    """Human approval step with multiple approval types."""
    if approval_type is None:
        print(f"=== HUMAN APPROVAL REQUIRED ===")
        print(f"Value: {value}")
        print("Approval options:")
        print("1. approve - Process normally")
        print("2. reject - Stop processing")
        print("3. modify - Request value modification")
        print("4. escalate - Send to supervisor")
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice == "1":
                approval_type = "approve"
            elif choice == "2":
                approval_type = "reject"
            elif choice == "3":
                approval_type = "modify"
            elif choice == "4":
                approval_type = "escalate"
            else:
                approval_type = "reject"  # Default to reject for invalid input
        except (EOFError, KeyboardInterrupt):
            approval_type = "reject"
    
    return {
        "value": value,
        "approval_type": approval_type,
        "approved": approval_type == "approve"
    }

# Human review with comments
def human_review_step(value: int, review_data: Optional[dict] = None) -> dict:
    """Human review step that captures comments and feedback."""
    if review_data is None:
        print(f"=== HUMAN REVIEW REQUIRED ===")
        print(f"Value to review: {value}")
        
        try:
            approved = input("Approve this value? (y/n): ").lower().strip() in ['y', 'yes']
            comments = input("Any comments or feedback? (optional): ").strip()
            priority = input("Priority level (low/medium/high): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            approved = False
            comments = "Review cancelled by user"
            priority = "low"
        
        review_data = {
            "approved": approved,
            "comments": comments,
            "priority": priority
        }
    
    return {
        "value": value,
        "review_data": review_data,
        "status": "reviewed"
    }

# Conditional human intervention
def conditional_human_step(value: int, auto_threshold: int = 50) -> dict:
    """Step that only requires human input for values above a threshold."""
    if value <= auto_threshold:
        # Auto-approve low values
        return {
            "value": value,
            "auto_approved": True,
            "human_reviewed": False
        }
    else:
        # Require human review for high values
        print(f"=== HUMAN REVIEW REQUIRED (Value {value} exceeds threshold {auto_threshold}) ===")
        
        try:
            approved = input("Approve this high-value item? (y/n): ").lower().strip() in ['y', 'yes']
            reason = input("Reason for approval/rejection: ").strip()
        except (EOFError, KeyboardInterrupt):
            approved = False
            reason = "Review cancelled by user"
        
        return {
            "value": value,
            "auto_approved": False,
            "human_reviewed": True,
            "approved": approved,
            "reason": reason
        }
`}/>

## Integration with external systems

<CodeBlock language="python" code={`# Webhook-based human approval
import json
from flask import Flask, request

app = Flask(__name__)

# Store pending approvals
pending_approvals = {}

def webhook_human_approval(value: int, approval_id: Optional[str] = None) -> dict:
    """Human approval step that waits for webhook confirmation."""
    if approval_id is None:
        # Generate approval ID and store pending request
        import uuid
        approval_id = str(uuid.uuid4())
        pending_approvals[approval_id] = {
            "value": value,
            "status": "pending"
        }
        
        print(f"=== WEBHOOK APPROVAL REQUIRED ===")
        print(f"Approval ID: {approval_id}")
        print(f"Value: {value}")
        print(f"Webhook URL: /approve/{approval_id}")
        print("Waiting for webhook confirmation...")
        
        # In a real implementation, this would wait for the webhook
        # For demonstration, we'll simulate it
        return {
            "value": value,
            "approval_id": approval_id,
            "status": "pending"
        }
    else:
        # Check if approval was received
        if approval_id in pending_approvals:
            approval_data = pending_approvals[approval_id]
            if approval_data["status"] == "approved":
                return {
                    "value": value,
                    "approval_id": approval_id,
                    "status": "approved",
                    "human_reviewed": True
                }
            elif approval_data["status"] == "rejected":
                return {
                    "value": value,
                    "approval_id": approval_id,
                    "status": "rejected",
                    "human_reviewed": True
                }
        
        # Still pending
        return {
            "value": value,
            "approval_id": approval_id,
            "status": "pending"
        }

@app.route('/approve/<approval_id>', methods=['POST'])
def approve(approval_id):
    """Webhook endpoint for human approval."""
    if approval_id in pending_approvals:
        data = request.get_json()
        action = data.get('action', 'reject')  # approve or reject
        
        pending_approvals[approval_id]['status'] = action
        
        return jsonify({
            "status": "success",
            "approval_id": approval_id,
            "action": action
        })
    
    return jsonify({"status": "error", "message": "Approval ID not found"}), 404

# Create workflow with webhook-based approval
webhook_human_workflow = (
    Workflow(name="webhook-human-workflow")
    .step(step1)
    .step(webhook_human_approval)
    .link("step1", "webhook_human_approval")
)
`}/>

## Key differences from Mastra

1. **Workflow Suspension**:
   - **Mastra**: Built-in `suspend()` method with complex resume schema
   - **Timbal**: Custom implementation using external systems or user input

2. **Resume Handling**:
   - **Mastra**: Complex `resumeData` and `resumeSchema` system
   - **Timbal**: Simple parameter passing and external input handling

3. **Workflow Structure**:
   - **Mastra**: `.then(step1).then(step2).commit()` with complex step definitions
   - **Timbal**: Simple `.step(step1).step(step2).link()` with custom suspension logic

4. **Human Interaction**:
   - **Mastra**: Framework-managed human interaction system
   - **Timbal**: Flexible integration with any human interaction system

5. **Implementation**:
   - **Mastra**: Complex async execution with built-in suspension
   - **Timbal**: Simple Python functions with custom suspension logic

The Timbal approach provides:
- **Flexibility**: Integrate with any human interaction system (UI, API, webhook)
- **Simplicity**: Use familiar Python patterns for workflow control
- **Customization**: Implement human-in-the-loop logic exactly as needed
- **Integration**: Easy to connect with existing approval systems
- **Scalability**: Can handle complex approval workflows and multiple stakeholders

This approach gives you the power to create sophisticated human-in-the-loop workflows while maintaining the simplicity and flexibility of the Timbal framework.
