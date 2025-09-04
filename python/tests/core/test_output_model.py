import json

from pydantic import BaseModel, Field
from timbal import Agent


class TestOutputModel:
    """Test Output Model functionality."""

    async def test_simple_output_model(self):
        """Test that the force_output_model tool is properly created and used."""
        """Check if a non specified term is also included in the answer."""

        class SimpleResponse(BaseModel):
            answer: str = Field(..., description="The answer to the question")
            confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level from 0.0 to 1.0")

        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            output_model=SimpleResponse
        )
        
        # Check the tool was added
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "output_model_tool"
        assert agent.tools[0].params_model == SimpleResponse
        
        user_input = "What is the capital of France?"
        response = await agent(prompt=user_input).collect()
        
        try:
            response_str = response.output.content[0].text
            response_data = json.loads(response_str)
            response_obj = SimpleResponse(**response_data)
            
            assert isinstance(response_obj.answer, str)
            assert isinstance(response_obj.confidence, float)
            assert 0.0 <= response_obj.confidence <= 1.0
            
        except Exception as e:
            assert False, f"Failed to parse structured response: {e}"
    

    async def test_nested_output_models(self):
        """Test that the output model is used correctly."""

        class Ingredient(BaseModel):
            name: str = Field(..., description="Name of the ingredient")
            amount: float = Field(..., description="Amount of the ingredient")
            unit: str = Field(..., description="Unit of the ingredient")

        class Recipe(BaseModel):
            ingredients: list[Ingredient] = Field(..., description="List of ingredients")
            total_time: float = Field(..., description="Total time in minutes")
            steps: list[str] = Field(..., description="Steps to follow")

        agent = Agent(
            name="output_model_agent",
            model="openai/gpt-4o-mini",
            output_model=Recipe)
        
        user_input = "I want to make lasagna, can you generate a lasagna recipe for me?"
        response = await agent(prompt=user_input).collect()

        try:
            recipe_str = response.output.content[0].text
            recipe = json.loads(recipe_str)
            recipe_obj = Recipe(**recipe)

        except Exception as e:
            assert False, f"Failed to parse the answer: {e}"

        assert recipe_obj.ingredients
        assert recipe_obj.total_time
        assert recipe_obj.steps

    
    async def test_followup_question_from_output(self):
        """Test that the agent can answer a follow-up question based on its own Recipe output
        without returning the raw schema.
        """

        class Ingredient(BaseModel):
            name: str = Field(..., description="Name of the ingredient")
            amount: float = Field(..., description="Amount of the ingredient")
            unit: str = Field(..., description="Unit of the ingredient")

        class Recipe(BaseModel):
            ingredients: list[Ingredient] = Field(..., description="List of ingredients")
            total_time: float = Field(..., description="Total time in minutes")
            steps: list[str] = Field(..., description="Steps to follow")

        agent = Agent(
            name="recipe_followup_agent",
            model="openai/gpt-4o-mini",
            output_model=Recipe
        )

        # First query: generate recipe
        user_input = "Give me a lasagna recipe."
        response = await agent(prompt=user_input).collect()

        try:
            recipe_str = response.output.content[0].text
            print(recipe_str)
            recipe_data = json.loads(recipe_str)
            recipe_obj = Recipe(**recipe_data)
        except Exception as e:
            assert False, f"Failed to parse initial recipe: {e}"

        # Second query (question based on the output)
        assert recipe_obj.ingredients
        assert recipe_obj.total_time > 0
        assert recipe_obj.steps

        followup_input = f"For how many people this recipe is suitable?"
        followup_response = await agent(prompt=followup_input).collect()
        followup_str = followup_response.output.content[0].text
        print("ANSWER", followup_str)

        try:
            json.loads(followup_str)
            assert False, "Follow-up response was JSON, but should have been plain text."
        except json.JSONDecodeError:
            assert isinstance(followup_str, str)
            assert len(followup_str.strip()) > 0

        # Third query (structured format again)
        user_input = "Adapt the recipe for two more people."
        response = await agent(prompt=user_input).collect()

        try:
            recipe_str = response.output.content[0].text
            print(recipe_str)
            recipe_data = json.loads(recipe_str)
            recipe_obj = Recipe(**recipe_data)
        except Exception as e:
            assert False, f"Failed to parse initial recipe: {e}"


    async def test_optional_fields_and_defaults(self):
        """Test output model with optional fields and default values."""

        class OutputModel(BaseModel):
            required_field: str = Field(..., description="This field is required")
            optional_field: str | None = Field(None, description="This field is optional")
            default_field: str = Field(default="default_value", description="Field with default")
            optional_with_default: int | None = Field(default=42, description="Optional with default")

        agent = Agent(
            name="flexible_agent",
            model="openai/gpt-4o-mini",
            output_model=OutputModel
        )
        
        user_input = "Generate data with only the required field set to 'test_value'"
        response = await agent(prompt=user_input).collect()
        
        try:
            response_str = response.output.content[0].text
            response_data = json.loads(response_str)
            obj = OutputModel(**response_data)
            
            assert obj.required_field == "test_value"
            assert obj.optional_field is None
            assert obj.default_field == "default_value"
            assert obj.optional_with_default == 42
            
        except Exception as e:
            assert False, f"Failed to parse flexible output model response: {e}"


    async def test_output_model_with_other_tools(self):
        """Test that output model works alongside other tools."""

        class CalculationResult(BaseModel):
            expression: str = Field(..., description="The mathematical expression")
            result: float = Field(..., description="The calculated result")
            method: str = Field(..., description="How the calculation was performed")

        def add(a: int, b: int) -> int:
            return a + b
        
        def multiply(a: int, b: int) -> int:
            return a * b

        agent = Agent(
            name="math_agent",
            model="openai/gpt-4.1",
            tools=[add, multiply],
            output_model=CalculationResult
        )
        assert len(agent.tools) == 3
        user_input = "Calculate 15 * 4 + 10"
        response = await agent(prompt=user_input).collect()
        
        try:
            response_str = response.output.content[0].text
            response_data = json.loads(response_str)
            result = CalculationResult(**response_data)

            assert isinstance(result.expression, str)
            assert isinstance(result.result, float)
            assert isinstance(result.method, str)
            
        except Exception as e:
            assert False, f"Failed to parse calculation result: {e}"


    async def test_output_model_with_list_constraints(self):
        """Test output model with list constraints and validation."""

        from pydantic import BaseModel, Field

        class ListOutput(BaseModel):
            numbers: list[int] = Field(..., min_length=2, max_length=5, description="List of 2-5 integers")
            tags: list[str] = Field(default_factory=list, description="List of string tags")

        agent = Agent(
            name="list_agent",
            model="openai/gpt-4o-mini",
            output_model=ListOutput
        )
        user_input = "Generate a list of 3 random numbers between 1 and 10"
        response = await agent(prompt=user_input).collect()
        
        try:
            response_str = response.output.content[0].text
            response_data = json.loads(response_str)
            result = ListOutput(**response_data)
            
            assert isinstance(result.numbers, list)
            assert 2 <= len(result.numbers) <= 5
            assert all(isinstance(n, int) for n in result.numbers)
            assert isinstance(result.tags, list)
            
        except Exception as e:
            assert False, f"Failed to parse list output model response: {e}"

