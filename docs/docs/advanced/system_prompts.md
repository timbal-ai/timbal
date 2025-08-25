---
title: Dynamic System Prompts
sidebar: 'docsSidebar'
---

# Dynamic System Prompts

Dynamic system prompts allow you to adapt LLM behavior based on context, input data, or execution state. Timbal v2 offers multiple strategies for creating powerful and dynamic system prompts.

## Variable Interpolation

System prompts can include variables that are resolved dynamically:

```python
from timbal.core_v2 import Agent
from timbal.types import Field

def create_dynamic_agent(user_role: str, company_context: str):
    # System prompt with variable interpolation
    system_prompt = f"""
    You are an assistant specialized in {user_role}.
    
    Company context:
    {company_context}
    
    Always respond in English and adapt to the specific role.
    """
    
    return Agent(
        model="gpt-4o-mini",
        system_prompt=system_prompt
    )

# Use the agent with specific context
agent = create_dynamic_agent(
    user_role="data analyst",
    company_context="We are a fintech startup working with AI"
)
```

## Data-Driven System Prompts

Prompts can be generated dynamically based on input data:

```python
from timbal.core_v2 import Agent
from timbal.types import Field

async def analyze_data_agent(data_type: str, data_sample: str):
    # Generate system prompt based on data type
    if data_type == "financial":
        system_prompt = f"""
        You are an expert financial analyst. Analyze the following data:
        
        {data_sample}
        
        Provide insights on trends, risks, and opportunities.
        """
    elif data_type == "customer":
        system_prompt = f"""
        You are a customer experience specialist. Analyze:
        
        {data_sample}
        
        Identify behavioral patterns and suggest improvements.
        """
    else:
        system_prompt = f"""
        Analyze this {data_type} data:
        
        {data_sample}
        
        Provide a detailed analysis and recommended actions.
        """
    
    return Agent(
        model="gpt-4o-mini",
        system_prompt=system_prompt
    )
```

## Conditional Prompts with Hooks

Hooks allow you to modify system prompts based on execution context:

```python
from timbal.core_v2 import Agent
from timbal.state import get_run_context

async def adaptive_system_prompt(input_dict):
    """Hook that adapts the system prompt based on context."""
    run_context = get_run_context()
    
    # Get context information
    user_level = run_context.data.get('user_level', 'beginner')
    language = run_context.data.get('language', 'english')
    
    # Generate adapted prompt
    if user_level == 'expert':
        system_prompt = """
        You are a technical expert. Provide detailed and technical analyses.
        Include code, diagrams, and technical references when appropriate.
        """
    else:
        system_prompt = """
        You are a patient instructor. Explain concepts clearly and simply.
        Avoid technical jargon and provide practical examples.
        """
    
    # Adapt language
    if language == 'spanish':
        system_prompt += "\nAlways respond in Spanish."
    else:
        system_prompt += "\nAlways respond in English."
    
    # Update the system prompt
    input_dict['system_prompt'] = system_prompt

# Agent with adaptive system prompt
agent = Agent(
    model="gpt-4o-mini",
    pre_hook=adaptive_system_prompt
)

# Execute with specific context
context = RunContext()
context.data['user_level'] = 'expert'
context.data['language'] = 'spanish'

result = await agent(
    prompt="Explain machine learning",
    context=context
).collect()
```

## Prompt Templates

Create reusable templates for different use cases:

```python
from string import Template

class PromptTemplates:
    ANALYST = Template("""
    You are an expert data analyst with specialization in $domain.
    
    Objective: $objective
    Context: $context
    
    Provide a detailed analysis following these steps:
    1. Executive summary
    2. Detailed analysis
    3. Conclusions and recommendations
    4. Key metrics
    """)
    
    COACH = Template("""
    You are a personalized coach for $role.
    
    User level: $user_level
    Goals: $goals
    
    Adapt to the level and provide practical advice and motivation.
    """)
    
    CREATIVE = Template("""
    You are a creative creator specialized in $creative_domain.
    
    Style: $style
    Tone: $tone
    
    Create original and ingenious content that fits the specified parameters.
    """)

# Use templates
def create_analyst_agent(domain: str, objective: str, context: str):
    system_prompt = PromptTemplates.ANALYST.substitute(
        domain=domain,
        objective=objective,
        context=context
    )
    
    return Agent(
        model="gpt-4o-mini",
        system_prompt=system_prompt
    )

# Usage example
agent = create_analyst_agent(
    domain="e-commerce",
    objective="Analyze customer purchase behavior",
    context="We have 6 months of online sales data"
)
```

## Multi-language Prompts

Manage system prompts in multiple languages:

```python
from timbal.core_v2 import Agent

class MultilingualPrompts:
    PROMPTS = {
        'english': {
            'analyst': "You are an expert analyst. Analyze the data and provide detailed insights.",
            'teacher': "You are a patient teacher. Explain concepts clearly and didactically.",
            'creative': "You are a creative creator. Generate original and ingenious content."
        },
        'spanish': {
            'analyst': "Eres un analista experto. Analiza los datos y proporciona insights detallados.",
            'teacher': "Eres un profesor paciente. Explica conceptos de manera clara y didáctica.",
            'creative': "Eres un creador creativo. Genera contenido original e ingenioso."
        },
        'french': {
            'analyst': "Vous êtes un analyste expert. Analysez les données et fournissez des insights détaillés.",
            'teacher': "Vous êtes un enseignant patient. Expliquez les concepts clairement et didactiquement.",
            'creative': "Vous êtes un créateur créatif. Générez du contenu original et ingénieux."
        }
    }
    
    @classmethod
    def get_prompt(cls, language: str, role: str) -> str:
        return cls.PROMPTS.get(language, cls.PROMPTS['english']).get(role, '')

# Multi-language agent
def create_multilingual_agent(language: str, role: str):
    system_prompt = MultilingualPrompts.get_prompt(language, role)
    
    return Agent(
        model="gpt-4o-mini",
        system_prompt=system_prompt
    )

# Usage examples
english_analyst = create_multilingual_agent('english', 'analyst')
spanish_teacher = create_multilingual_agent('spanish', 'teacher')
french_creative = create_multilingual_agent('french', 'creative')
```

## Context-Aware Prompts

Prompts can adapt based on conversation history or current state:

```python
from timbal.core_v2 import Agent
from timbal.state import get_run_context

async def context_aware_prompt(input_dict):
    """Hook that adapts the prompt based on conversation context."""
    run_context = get_run_context()
    
    # Get conversation history
    conversation_history = run_context.data.get('conversation_history', [])
    
    # Analyze conversation context
    if len(conversation_history) > 5:
        # Long conversation - switch to summary mode
        input_dict['system_prompt'] += "\n\nThe conversation is long. Provide concise and direct responses."
    
    # Detect conversation topic
    if any('error' in msg.lower() for msg in conversation_history[-3:]):
        input_dict['system_prompt'] += "\n\nI detect there have been errors. Provide practical solutions."
    
    # Adapt based on time of day
    from datetime import datetime
    hour = datetime.now().hour
    if 22 <= hour or hour <= 6:
        input_dict['system_prompt'] += "\n\nIt's late. Keep responses brief and direct."

# Agent with dynamic context
agent = Agent(
    model="gpt-4o-mini",
    system_prompt="You are a helpful and friendly assistant.",
    pre_hook=context_aware_prompt
)
```

## Complete Example: Adaptive Prompt System

```python
from timbal.core_v2 import Agent
from timbal.state import RunContext
from timbal.types import Field

class AdaptivePromptSystem:
    def __init__(self):
        self.base_prompts = {
            'technical': "You are a technical expert. Provide detailed and technical solutions.",
            'business': "You are a business consultant. Provide strategic and practical insights.",
            'creative': "You are a creative creator. Generate original and innovative ideas."
        }
    
    async def adaptive_hook(self, input_dict):
        run_context = get_run_context()
        
        # Determine query type
        user_input = input_dict.get('prompt', '')
        if any(word in user_input.lower() for word in ['code', 'technical', 'programming']):
            prompt_type = 'technical'
        elif any(word in user_input.lower() for word in ['business', 'strategy', 'market']):
            prompt_type = 'business'
        else:
            prompt_type = 'creative'
        
        # Adapt the prompt
        base_prompt = self.base_prompts[prompt_type]
        
        # Add specific context
        if run_context.data.get('user_expertise') == 'beginner':
            base_prompt += "\n\nThe user is a beginner. Explain concepts simply."
        
        input_dict['system_prompt'] = base_prompt

# Create adaptive agent
prompt_system = AdaptivePromptSystem()
agent = Agent(
    model="gpt-4o-mini",
    pre_hook=prompt_system.adaptive_hook
)

# Execute with context
context = RunContext()
context.data['user_expertise'] = 'beginner'

result = await agent(
    prompt="Explain how machine learning works",
    context=context
).collect()
```

These examples show how dynamic system prompts enable the creation of more intelligent and adaptive agents that behave according to the specific context of each interaction.
