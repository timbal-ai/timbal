---
title: Advanced Flow Concepts
sidebar: 'docsSidebar'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Advanced Flow Concepts

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Build complex workflows with conditional branching, tool integration, nested flows, and advanced debugging.
</h2>

---

## Conditional Branching

Flows can make decisions about which path to take using conditional links.

This enables dynamic, data-driven workflows—where the next step depends on the result of a previous step.

### Example: One Conditional Link

Suppose you want to review a document, and if it contains sensitive information, notify compliance; otherwise, archive it.

<CodeBlock language="python" code={`from timbal import Flow

def check_vip(user):
    return user["is_vip"]

def send_discount_email(user):
    # In a real app, this would send an email
    return f"Discount email sent to {user['email']}!"

flow = (
    Flow()
    .add_step("check", check_vip)
    .add_step("send_email", send_discount_email)
    # Only send email if user is VIP
    .add_link("check", "send_email", condition="{{check.return}} == True")
    .set_input("check.user", "user_info")
    .set_data_map("send_email.user", "check.user")
    .set_output("send_email.return", "result")
)

async def main():
    vip_user = {"email": "alice@example.com", "is_vip": True}
    regular_user = {"email": "bob@example.com", "is_vip": False}

    result = await flow.complete(user_info=vip_user)
    # Output: Discount email sent to alice@example.com!

    result = await flow.complete(user_info=regular_user)
    # Output: None
`}/>

**How it works**:
- The check step checks if the user is a VIP.
- If is_vip is True, the flow continues to send_email.
- If not, the flow ends and no email is sent (no result is produced).

---

## Tool Integration: 

LLMs in flows can call tools (functions or APIs) as part of their reasoning.  
You can link LLM steps to tool steps using `is_tool=True` and `is_tool_result=True`:

### Example: LLM with Weather Tool

Suppose you want your LLM to answer general questions, but if the user asks about the weather, it should call a weather tool to get the latest information.

<CodeBlock language="python" code={`from timbal import Flow

def get_weather(city):
    # Simulate a weather API call 
    return f"The weather in {city} is sunny and 25°C."

flow = (
    Flow()
    .add_llm("llm", model="gpt-4.1-nano", memory_id="llm")
    .set_input("llm.prompt", "user_question")
    .add_step("weather_tool", get_weather) 
    # Link the LLM to the weather tool as a callable function
    .add_link("llm", "weather_tool", is_tool=True)
    # Setting the same memory as the first LLM
    .add_llm("llm2", model="gpt-4.1-nano", memory_id="llm")
    .add_link("weather_tool", "llm2", is_tool_result=True)
    # Return the tool result to the LLM for final response
    .set_output("llm2.return", "response")
)
    
async def main(): 
    result = await flow.complete(user_question="What's the weather in Paris?")
    print(result.output["response"].content[0].text)
    # Output: The weather in Paris is sunny and 25°C.`}/>

**Key Points**:
- `memory_id`: Enables memory for LLMs. Using the same `memory_id` for both LLMs means they share context and conversation history.
- `is_tool` / `is_tool_result`: Allows the LLM to call a tool and then use the tool’s result in a follow-up LLM step.
- Chaining with memory: The second LLM (**llm2**) can generate a more informed, context-aware response because it shares memory with the first LLM.

---

## Nesting Flows

You can use a flow as a step inside another flow, enabling modular, reusable workflow components.

### Example: Email Processing and Summarization

Suppose you want to process incoming emails by first extracting the main content and then summarizing it with an LLM. 

You can define a reusable subflow for the extraction and summarization, and then use it as a step in your main workflow.

<CodeBlock language="python" code={`from timbal import Flow

def validate_order(order):
    # Simulate checking if the order is valid
    if order["quantity"] > 0 and order["item"] in ["apple", "banana"]:
        return {"valid": True, "item": order["item"], "quantity": order["quantity"]}
    else:
        return {"valid": False, "reason": "Invalid item or quantity"}

def confirm_order(validated):
    if validated["valid"]:
        return f"Order confirmed: {validated['quantity']} {validated['item']}(s)."
    else:
        return f"Order failed: {validated['reason']}"

# Define the reusable subflow for validation
validation_flow = (
    Flow()
    .add_step("validate", validate_order)
    .set_input("validate.order", "order")
    .set_output("validate.return", "validated")
)

# Main flow uses the subflow and then confirms the order
main_flow = (
    Flow()
    .add_step("get_order", lambda: {"item": "apple", "quantity": 3})
    .add_step("validate_order", validation_flow)
    .add_step("confirm", confirm_order)
    .set_data_map("validate_order.order", "get_order.return")
    .set_data_map("confirm.validated", "validate_order.return.validated")
    .set_output("confirm.return", "confirmation")
)

async def main():
    result = await main_flow.complete()
    print(result.output["confirmation"])
    # Output: Order confirmed: 3 apple(s).`}/>


<div className="log-step-static">
StartEvent(..., path='flow', ...)
</div>
<div className="log-step-static">
StartEvent(..., path='flow.get_order', ...)
</div>
<div className="log-step-static">
StartEvent(..., path='flow.validate_order', ...)
</div>
<details className="log-step-collapsible">
<summary>
OutputEvent(...,path='flow.get_order', ...)
)
</summary>
<CodeBlock language="bash" code={`OutputEvent(
    ..., 
    path='flow.get_order',
    input={},
    output={'item': 'apple', 'quantity': 3}, 
    ...
)`}/>
</details>

<div className="log-step-static">
StartEvent(..., path='flow.validate_order', ...)
</div>
<div className="log-step-static">
StartEvent(..., path='flow.validate_order.validate', ...)
</div>
<details className="log-step-collapsible">
<summary>
OutputEvent(..., path='flow.validate_order.validate', ...)
</summary>
<CodeBlock language="bash" code={`OutputEvent(
    ...,
    path='flow',
    input={},
    output={'confirmation': 'Order confirmed: 3 apple(s).'},
    ...
)`}/>
</details>


---

## Debugging and Visualization

Use Timbal's built-in tools to visualize and debug your flows.  
You can inspect the execution graph, step outputs, and data mappings to understand and optimize your workflow.

---

For more, see the [Flows Overview](/flows) and [Examples](/examples).