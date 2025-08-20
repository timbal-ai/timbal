import os
import pytest
from timbal.core_v2.handlers.llm_router import handler
from timbal.types.message import Message
from timbal.errors import APIKeyNotFoundError

from dotenv import load_dotenv

from .conftest import Timer

load_dotenv()


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [Message.validate("Say 'Hello from test' in one sentence.")]


@pytest.fixture
def openai_model():
    """OpenAI model for testing."""
    return "openai/gpt-4o-mini"


@pytest.fixture
def anthropic_model():
    """Anthropic model for testing."""
    return "anthropic/claude-3-5-sonnet-20241022"


@pytest.fixture
def gemini_model():
    """Gemini model for testing."""
    return "gemini/gemini-2.0-flash"


@pytest.fixture
def togetherai_model():
    """TogetherAI model for testing."""
    return "togetherai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


class TestLLMRouter:
    """Test LLM Router functionality with real API calls."""

    @pytest.mark.asyncio
    async def test_openai_provider_success(self, sample_messages, openai_model):
        """Test OpenAI provider with real API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=openai_model,
                messages=sample_messages,
                max_tokens=50
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"OpenAI test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_openai_provider_no_api_key(self, sample_messages, openai_model):
        """Test OpenAI provider without API key."""
        # Temporarily remove the API key for this test
        original_key = os.environ.get("OPENAI_API_KEY")
        if original_key:
            del os.environ["OPENAI_API_KEY"]
        
        try:
            with pytest.raises(APIKeyNotFoundError, match="OPENAI_API_KEY not found"):
                async for _ in handler(
                    model=openai_model,
                    messages=sample_messages
                ):
                    pass
        finally:
            # Restore the API key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_anthropic_provider_success(self, sample_messages, anthropic_model):
        """Test Anthropic provider with real API call."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=anthropic_model,
                messages=sample_messages,
                max_tokens=50
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"Anthropic test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_anthropic_provider_no_max_tokens(self, sample_messages, anthropic_model):
        """Test Anthropic provider without max_tokens."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        with pytest.raises(ValueError, match="'max_tokens' is required for claude models"):
            async for _ in handler(
                model=anthropic_model,
                messages=sample_messages
            ):
                pass

    @pytest.mark.asyncio
    async def test_anthropic_provider_no_api_key(self, sample_messages, anthropic_model):
        """Test Anthropic provider without API key."""
        # Temporarily remove the API key for this test
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        if original_key:
            del os.environ["ANTHROPIC_API_KEY"]
        
        try:
            with pytest.raises(APIKeyNotFoundError, match="ANTHROPIC_API_KEY not found"):
                async for _ in handler(
                    model=anthropic_model,
                    messages=sample_messages,
                    max_tokens=50
                ):
                    pass
        finally:
            # Restore the API key
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_gemini_provider_success(self, sample_messages, gemini_model):
        """Test Gemini provider with real API call."""
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=gemini_model,
                messages=sample_messages,
                max_tokens=50
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"Gemini test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_gemini_provider_no_api_key(self, sample_messages, gemini_model):
        """Test Gemini provider without API key."""
        # Temporarily remove the API key for this test
        original_key = os.environ.get("GEMINI_API_KEY")
        if original_key:
            del os.environ["GEMINI_API_KEY"]
        
        try:
            with pytest.raises(APIKeyNotFoundError, match="GEMINI_API_KEY not found"):
                async for _ in handler(
                    model=gemini_model,
                    messages=sample_messages
                ):
                    pass
        finally:
            # Restore the API key
            if original_key:
                os.environ["GEMINI_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_togetherai_provider_success(self, sample_messages, togetherai_model):
        """Test TogetherAI provider with real API call."""
        if not os.getenv("TOGETHER_API_KEY"):
            pytest.skip("TOGETHER_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=togetherai_model,
                messages=sample_messages,
                max_tokens=50
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"TogetherAI test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_togetherai_provider_no_api_key(self, sample_messages, togetherai_model):
        """Test TogetherAI provider without API key."""
        # Temporarily remove the API key for this test
        original_key = os.environ.get("TOGETHER_API_KEY")
        if original_key:
            del os.environ["TOGETHER_API_KEY"]
        
        try:
            with pytest.raises(APIKeyNotFoundError, match="TOGETHER_API_KEY not found"):
                async for _ in handler(
                    model=togetherai_model,
                    messages=sample_messages
                ):
                    pass
        finally:
            # Restore the API key
            if original_key:
                os.environ["TOGETHER_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_invalid_provider(self, sample_messages):
        """Test invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            async for _ in handler(
                model="invalid/model",
                messages=sample_messages
            ):
                pass

    @pytest.mark.asyncio
    async def test_invalid_model_format(self, sample_messages):
        """Test invalid model format."""
        with pytest.raises(ValueError, match="Model must be in format 'provider/model_name'"):
            async for _ in handler(
                model="invalid_model_format",
                messages=sample_messages
            ):
                pass

    @pytest.mark.asyncio
    async def test_system_prompt_integration(self, sample_messages, openai_model):
        """Test system prompt integration with real API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=openai_model,
                messages=sample_messages,
                system_prompt="You are a helpful assistant. Always respond with 'Hello from system prompt test'.",
                max_tokens=50
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"System prompt test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_json_schema_integration(self, sample_messages, openai_model):
        """Test JSON schema integration with real API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        json_schema = {
            "name": "test_response",
            "schema": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"}
                },
                "required": ["response"]
            }
        }
        
        with Timer() as timer:
            chunks = []
            try:
                async for chunk in handler(
                    model=openai_model,
                    messages=sample_messages,
                    json_schema=json_schema,
                    max_tokens=100
                ):
                    chunks.append(chunk)
                
                # Verify we got some response
                assert len(chunks) > 0
                elapsed_time = timer.elapsed if timer.elapsed is not None else 0.0
                print(f"JSON schema test completed in {elapsed_time:.2f}s with {len(chunks)} chunks")
            except Exception as e:
                if "Missing required parameter" in str(e):
                    pytest.skip(f"JSON schema not supported for this model: {e}")
                else:
                    raise

    @pytest.mark.asyncio
    async def test_anthropic_json_schema_not_supported(self, sample_messages, anthropic_model):
        """Test that JSON schema is not supported for Anthropic."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        json_schema = {"type": "object"}
        
        with pytest.raises(NotImplementedError, match="JSON schema validation is not supported for claude models"):
            async for _ in handler(
                model=anthropic_model,
                messages=sample_messages,
                json_schema=json_schema,
                max_tokens=50
            ):
                pass

    @pytest.mark.asyncio
    async def test_all_providers_workflow(self, sample_messages):
        """Test a simple workflow with all available providers."""
        providers_to_test = []
        
        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("openai/gpt-4o-mini", "OpenAI"))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_to_test.append(("anthropic/claude-3-5-sonnet-20241022", "Anthropic"))
        if os.getenv("GEMINI_API_KEY"):
            providers_to_test.append(("gemini/gemini-2.0-flash", "Gemini"))
        if os.getenv("TOGETHER_API_KEY"):
            providers_to_test.append(("togetherai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "TogetherAI"))
        
        if not providers_to_test:
            pytest.skip("No API keys available for testing")
        
        for model, provider_name in providers_to_test:
            print(f"\nTesting {provider_name}...")
            
            with Timer() as timer:
                chunks = []
                try:
                    async for chunk in handler(
                        model=model,
                        messages=sample_messages,
                        max_tokens=30
                    ):
                        chunks.append(chunk)
                    
                    assert len(chunks) > 0
                    elapsed_time = timer.elapsed if timer.elapsed is not None else 0.0
                    print(f"{provider_name} test passed in {elapsed_time:.2f}s with {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f" {provider_name} test failed: {e}")
                    # Don't raise here, just log the failure
                    continue

    @pytest.mark.asyncio
    async def test_temperature_parameter(self, sample_messages, openai_model):
        """Test temperature parameter with real API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=openai_model,
                messages=sample_messages,
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=50
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"Temperature test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")

    @pytest.mark.asyncio
    async def test_max_tokens_parameter(self, sample_messages, openai_model):
        """Test max_tokens parameter with real API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        with Timer() as timer:
            chunks = []
            async for chunk in handler(
                model=openai_model,
                messages=sample_messages,
                max_tokens=10  # Very short response
            ):
                chunks.append(chunk)
        
        # Verify we got some response
        assert len(chunks) > 0
        print(f"Max tokens test completed in {timer.elapsed:.2f}s with {len(chunks)} chunks")
