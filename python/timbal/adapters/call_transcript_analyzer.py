"""
Function to analyze the transcript of a customer satisfaction survey call.
"""

import json
from typing import Dict, Optional, Any

from timbal import Agent
from timbal.state.savers import InMemorySaver


async def analyze_transcript(agent_state: Dict) -> Optional[Dict[str, Any]]:
    """
    Analyzes the transcript of a customer satisfaction survey call.
    
    Args:
        agent_state: The agent's execution state, including conversation history.
        
    Returns:
        A dictionary with the structured analysis of the call, or None if analysis fails.
    """
    
    # Create the analyzer agent
    analyzer_agent = Agent(
        model="gpt-4o-mini",
        system_prompt="""
        You are an expert analyst of call transcripts for a customer satisfaction survey.
        Your task is to analyze the following transcript and extract key information.
        Provide your analysis ONLY in a valid JSON format. Do not add any other text.
        The survey has two questions: 1. A satisfaction rating from 1 to 5. 2. The reason for that rating.
        
        The JSON output must conform to the following schema:
        {
          "type": "object",
          "properties": {
            "survey_completed": {
              "type": "boolean",
              "description": "True if the customer answered both survey questions, false otherwise."
            },
            "satisfaction_rating": {
              "type": "integer",
              "description": "The satisfaction rating from 1 to 5 provided by the customer. Null if no rating was given."
            },
            "reason_for_rating": {
              "type": "string",
              "description": "A brief summary of the customer's stated reason for their rating. Null if no reason was provided."
            },
            "customer_response": {
              "type": "string",
              "enum": ["completed", "declined", "incomplete"],
              "description": "'completed' if they answered both questions. 'declined' if they refused to participate at any point. 'incomplete' if they started but didn't finish."
            },
            "call_outcome": {
              "type": "string",
              "enum": ["successful_survey", "declined", "no_rating_provided", "unclear"],
              "description": "'successful_survey' if a rating was provided. 'declined' if the user refused. 'no_rating_provided' if they agreed but didn't give a score. 'unclear' for other cases."
            }
          },
          "required": ["survey_completed", "satisfaction_rating", "reason_for_rating", "customer_response", "call_outcome"]
        }
        """,
        state_saver=InMemorySaver(),
        stream=False
    )
    
    # Extract conversation text
    conversation_text = ""
    conversation_history = agent_state.get("conversation_history", [])
    
    for entry in conversation_history:
        speaker = "CLIENTE" if entry.get("speaker") == "user" else "AGENTE"
        text = entry.get("text", "")
        conversation_text += f"{speaker}: {text}\n"
    
    if not conversation_text.strip():
        return None
    
    # Analyze with the detector agent
    prompt = f"Analyze this customer satisfaction survey call and extract the required information:\n\n{conversation_text}"
    
    # Run the agent and get the response.
    # With stream=False, run() still returns an async generator that yields a final output event.
    final_event = None
    async for event in analyzer_agent.run(prompt=prompt):
        final_event = event

    result_text = ""
    # The output from a non-streaming agent run is an object with a 'content' attribute.
    if final_event and hasattr(final_event, 'output') and final_event.output:
        # The output from a non-streaming agent run is an object with a `content` attribute.
        if hasattr(final_event.output, 'content'):
            content_list = final_event.output.content
            # The content is a list of blocks, usually one TextBlock.
            if isinstance(content_list, list) and content_list:
                first_block = content_list[0]
                # The block itself has a `text` attribute.
                if hasattr(first_block, 'text'):
                    result_text = first_block.text

    if not result_text.strip():
        return None
    
    # Parse JSON response
    try:
        # Clean up the response
        cleaned_result = result_text.strip()
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result.replace("```json", "", 1).strip()
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3].strip()
        
        parsed_json = json.loads(cleaned_result)
        
        if not isinstance(parsed_json, dict):
            return {"error": "Invalid response format from analyzer."}
            
        return parsed_json
        
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parsing error: {str(e)}"
        }