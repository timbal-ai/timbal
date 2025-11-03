"""Tests for LLM router retry logic."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic import (
    RateLimitError as AnthropicRateLimitError,
    APIStatusError as AnthropicAPIStatusError,
    APITimeoutError as AnthropicAPITimeoutError,
    APIConnectionError as AnthropicAPIConnectionError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
    APIStatusError as OpenAIAPIStatusError,
    APITimeoutError as OpenAIAPITimeoutError,
    APIConnectionError as OpenAIAPIConnectionError,
)

from timbal.core.llm_router import _retry_on_error


class TestRetryOnError:
    """Test the _retry_on_error helper function."""

    @pytest.mark.asyncio
    async def test_empty_stream_retries(self):
        """Test that empty streams are retried."""
        attempt_count = 0
        
        async def empty_stream():
            nonlocal attempt_count
            attempt_count += 1
            # Empty async generator - never yields
            return
            yield  # Make it a generator but never yield
        
        # In Python 3.7+, StopAsyncIteration raised from async generator becomes RuntimeError
        with pytest.raises((StopAsyncIteration, RuntimeError)):
            async for _ in _retry_on_error(empty_stream, max_retries=2, retry_delay=0.01, context="Test"):
                pass
        
        # Should have tried 3 times (initial + 2 retries)
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_successful_stream_no_retry(self):
        """Test that successful streams don't retry."""
        attempt_count = 0
        
        async def successful_stream():
            nonlocal attempt_count
            attempt_count += 1
            yield "chunk1"
            yield "chunk2"
        
        chunks = []
        async for chunk in _retry_on_error(successful_stream, max_retries=2, retry_delay=0.01, context="Test"):
            chunks.append(chunk)
        
        assert chunks == ["chunk1", "chunk2"]
        assert attempt_count == 1  # No retries needed

    @pytest.mark.asyncio
    async def test_rate_limit_error_retries(self):
        """Test that rate limit errors are retried."""
        attempt_count = 0
        
        async def rate_limited_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                # Fail first 2 attempts with rate limit
                raise OpenAIRateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
            
            # Succeed on 3rd attempt
            yield "success"
        
        chunks = []
        async for chunk in _retry_on_error(rate_limited_stream, max_retries=3, retry_delay=0.01, context="Test"):
            chunks.append(chunk)
        
        assert chunks == ["success"]
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self):
        """Test that timeout errors are retried."""
        attempt_count = 0
        
        async def timeout_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                raise AnthropicAPITimeoutError(request=MagicMock())
            
            yield "success"
        
        chunks = []
        async for chunk in _retry_on_error(timeout_stream, max_retries=2, retry_delay=0.01, context="Test"):
            chunks.append(chunk)
        
        assert chunks == ["success"]
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_connection_error_retries(self):
        """Test that connection errors are retried."""
        attempt_count = 0
        
        async def connection_error_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                raise OpenAIAPIConnectionError(request=MagicMock())
            
            yield "success"
        
        chunks = []
        async for chunk in _retry_on_error(connection_error_stream, max_retries=2, retry_delay=0.01, context="Test"):
            chunks.append(chunk)
        
        assert chunks == ["success"]
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_server_error_503_retries(self):
        """Test that 503 server errors are retried."""
        attempt_count = 0
        
        async def server_error_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                response = MagicMock()
                response.status_code = 503
                raise AnthropicAPIStatusError(
                    message="Service unavailable",
                    response=response,
                    body=None
                )
            
            yield "success"
        
        chunks = []
        async for chunk in _retry_on_error(server_error_stream, max_retries=2, retry_delay=0.01, context="Test"):
            chunks.append(chunk)
        
        assert chunks == ["success"]
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_server_error_500_retries(self):
        """Test that 500 server errors are retried."""
        attempt_count = 0
        
        async def server_error_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                response = MagicMock()
                response.status_code = 500
                raise OpenAIAPIStatusError(
                    message="Internal server error",
                    response=response,
                    body=None
                )
            
            yield "success"
        
        chunks = []
        async for chunk in _retry_on_error(server_error_stream, max_retries=2, retry_delay=0.01, context="Test"):
            chunks.append(chunk)
        
        assert chunks == ["success"]
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_client_error_400_no_retry(self):
        """Test that 400 client errors are NOT retried."""
        attempt_count = 0
        
        async def client_error_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            response = MagicMock()
            response.status_code = 400
            raise OpenAIAPIStatusError(
                message="Bad request",
                response=response,
                body=None
            )
            yield  # Make it a generator
        
        with pytest.raises(OpenAIAPIStatusError):
            async for _ in _retry_on_error(client_error_stream, max_retries=2, retry_delay=0.01, context="Test"):
                pass
        
        # Should fail immediately, no retries
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_401_no_retry(self):
        """Test that 401 auth errors are NOT retried."""
        attempt_count = 0
        
        async def auth_error_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            response = MagicMock()
            response.status_code = 401
            raise AnthropicAPIStatusError(
                message="Unauthorized",
                response=response,
                body=None
            )
            yield  # Make it a generator
        
        with pytest.raises(AnthropicAPIStatusError):
            async for _ in _retry_on_error(auth_error_stream, max_retries=2, retry_delay=0.01, context="Test"):
                pass
        
        # Should fail immediately, no retries
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        attempt_count = 0
        
        async def always_fails_stream():
            nonlocal attempt_count
            attempt_count += 1
            raise OpenAIRateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
            yield  # Make it a generator
        
        with pytest.raises(OpenAIRateLimitError):
            async for _ in _retry_on_error(always_fails_stream, max_retries=2, retry_delay=0.01, context="Test"):
                pass
        
        # Should try 3 times (initial + 2 retries)
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test that exponential backoff is applied."""
        attempt_count = 0
        delays = []
        
        async def rate_limited_stream():
            nonlocal attempt_count
            attempt_count += 1
            raise OpenAIRateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
        
        # Patch asyncio.sleep to capture delays
        original_sleep = asyncio.sleep
        
        async def mock_sleep(delay):
            delays.append(delay)
            await original_sleep(0.001)  # Actually sleep a tiny bit
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            with pytest.raises(OpenAIRateLimitError):
                async for _ in _retry_on_error(always_fails_stream, max_retries=3, retry_delay=1.0, context="Test"):
                    pass
        
        # Should have delays: 1.0, 2.0, 4.0 (exponential backoff)
        assert len(delays) == 3
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0

    @pytest.mark.asyncio
    async def test_first_chunk_then_error(self):
        """Test that errors after first chunk are not retried."""
        attempt_count = 0
        
        async def partial_stream():
            nonlocal attempt_count
            attempt_count += 1
            
            yield "chunk1"
            # Error after first chunk
            raise RuntimeError("Something went wrong mid-stream")
        
        chunks = []
        with pytest.raises(RuntimeError):
            async for chunk in _retry_on_error(partial_stream, max_retries=2, retry_delay=0.01, context="Test"):
                chunks.append(chunk)
        
        # Should have gotten first chunk
        assert chunks == ["chunk1"]
        # Should not retry (error happened after stream started)
        assert attempt_count == 1


# Helper for exponential backoff test
async def always_fails_stream():
    raise OpenAIRateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
    yield  # Make it a generator
