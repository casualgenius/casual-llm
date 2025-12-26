"""
Tests for embedding provider implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from casual_llm.config import ModelConfig, Provider
from casual_llm.embeddings import (
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    create_embedding_provider,
)
from casual_llm.usage import Usage


# Try to import OpenAI provider - may not be available
try:
    from casual_llm.embeddings import OpenAIEmbeddingProvider

    OPENAI_AVAILABLE = OpenAIEmbeddingProvider is not None
except ImportError:
    OPENAI_AVAILABLE = False


class TestEmbeddingModels:
    """Tests for EmbeddingData and EmbeddingResponse Pydantic models"""

    def test_embedding_data_valid(self):
        """Test valid EmbeddingData creation"""
        data = EmbeddingData(
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            index=0,
        )
        assert data.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert data.index == 0

    def test_embedding_data_index_validation(self):
        """Test that EmbeddingData index must be >= 0"""
        with pytest.raises(ValueError):
            EmbeddingData(
                embedding=[0.1, 0.2, 0.3],
                index=-1,  # Invalid: negative index
            )

    def test_embedding_response_valid(self):
        """Test valid EmbeddingResponse creation"""
        data = [
            EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
            EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
        ]
        response = EmbeddingResponse(
            data=data,
            model="test-model",
            dimensions=3,
            usage=None,
        )
        assert len(response.data) == 2
        assert response.model == "test-model"
        assert response.dimensions == 3
        assert response.usage is None

    def test_embedding_response_with_usage(self):
        """Test EmbeddingResponse with usage statistics"""
        usage = Usage(prompt_tokens=10, completion_tokens=0)
        response = EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
            model="test-model",
            dimensions=2,
            usage=usage,
        )
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 10

    def test_embedding_response_dimensions_validation(self):
        """Test that dimensions must be >= 0"""
        with pytest.raises(ValueError):
            EmbeddingResponse(
                data=[],
                model="test-model",
                dimensions=-1,  # Invalid: negative dimensions
                usage=None,
            )

    def test_embedding_response_empty_data(self):
        """Test EmbeddingResponse with empty data (valid for empty input)"""
        # Empty data with dimensions=0 is valid for empty input case
        response = EmbeddingResponse(
            data=[],
            model="test-model",
            dimensions=0,
            usage=None,
        )
        assert len(response.data) == 0
        assert response.model == "test-model"
        assert response.dimensions == 0


class TestEmbeddingProviderProtocol:
    """Tests for EmbeddingProvider protocol compatibility"""

    def test_protocol_defines_embed_method(self):
        """Test that protocol requires embed method"""
        # EmbeddingProvider is a Protocol with embed() and get_usage() methods
        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "get_usage")

    def test_ollama_provider_satisfies_protocol(self):
        """Test that OllamaEmbeddingProvider satisfies the protocol"""
        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        # Should have required methods
        assert hasattr(provider, "embed")
        assert hasattr(provider, "get_usage")
        assert callable(provider.embed)
        assert callable(provider.get_usage)

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
    def test_openai_provider_satisfies_protocol(self):
        """Test that OpenAIEmbeddingProvider satisfies the protocol"""
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="sk-test")
        # Should have required methods
        assert hasattr(provider, "embed")
        assert hasattr(provider, "get_usage")
        assert callable(provider.embed)
        assert callable(provider.get_usage)


class TestOllamaEmbeddingProvider:
    """Tests for OllamaEmbeddingProvider"""

    @pytest.fixture
    def provider(self):
        """Create an OllamaEmbeddingProvider instance for testing"""
        return OllamaEmbeddingProvider(
            model="nomic-embed-text",
            host="http://localhost:11434",
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_embed_single_text_success(self, provider):
        """Test successful single text embedding"""
        mock_response = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "prompt_eval_count": 5,
        }

        with patch.object(provider.client, "embed", new=AsyncMock(return_value=mock_response)):
            result = await provider.embed(["Hello, world!"])

            assert isinstance(result, EmbeddingResponse)
            assert len(result.data) == 1
            assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert result.data[0].index == 0
            assert result.model == "nomic-embed-text"
            assert result.dimensions == 5

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, provider):
        """Test successful batch embedding"""
        mock_response = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            "prompt_eval_count": 15,
        }

        with patch.object(provider.client, "embed", new=AsyncMock(return_value=mock_response)):
            texts = ["First", "Second", "Third"]
            result = await provider.embed(texts)

            assert isinstance(result, EmbeddingResponse)
            assert len(result.data) == 3
            assert result.data[0].embedding == [0.1, 0.2, 0.3]
            assert result.data[1].embedding == [0.4, 0.5, 0.6]
            assert result.data[2].embedding == [0.7, 0.8, 0.9]
            assert result.data[0].index == 0
            assert result.data[1].index == 1
            assert result.data[2].index == 2
            assert result.dimensions == 3

    @pytest.mark.asyncio
    async def test_embed_empty_input(self, provider):
        """Test handling of empty input list"""
        result = await provider.embed([])

        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 0
        assert result.model == "nomic-embed-text"
        assert result.dimensions == 0
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_embed_model_override(self, provider):
        """Test that model can be overridden per-call"""
        mock_response = {
            "embeddings": [[0.1, 0.2]],
            "prompt_eval_count": 3,
        }

        mock_embed = AsyncMock(return_value=mock_response)
        with patch.object(provider.client, "embed", new=mock_embed):
            result = await provider.embed(["Test"], model="different-model")

            # Verify the overridden model was passed
            call_kwargs = mock_embed.call_args.kwargs
            assert call_kwargs["model"] == "different-model"
            assert result.model == "different-model"

    @pytest.mark.asyncio
    async def test_usage_tracking(self, provider):
        """Test that usage statistics are tracked"""
        # Check initial state
        assert provider.get_usage() is None

        mock_response = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "prompt_eval_count": 10,
        }

        with patch.object(provider.client, "embed", new=AsyncMock(return_value=mock_response)):
            await provider.embed(["Test text"])

            # Verify usage was tracked
            usage = provider.get_usage()
            assert usage is not None
            assert isinstance(usage, Usage)
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 0
            assert usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_usage_tracking_no_tokens(self, provider):
        """Test usage when no token count is returned"""
        mock_response = {
            "embeddings": [[0.1, 0.2, 0.3]],
            # No prompt_eval_count in response
        }

        with patch.object(provider.client, "embed", new=AsyncMock(return_value=mock_response)):
            result = await provider.embed(["Test text"])

            # Usage should be None when not available
            assert provider.get_usage() is None
            assert result.usage is None

    @pytest.mark.asyncio
    async def test_dimensions_match_vector_length(self, provider):
        """Test that dimensions field matches actual embedding vector length"""
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mock_response = {
            "embeddings": [embedding_vector],
            "prompt_eval_count": 5,
        }

        with patch.object(provider.client, "embed", new=AsyncMock(return_value=mock_response)):
            result = await provider.embed(["Test"])

            assert result.dimensions == len(embedding_vector)
            assert result.dimensions == len(result.data[0].embedding)

    def test_host_trailing_slash_handling(self):
        """Test that trailing slashes are removed from host URL"""
        provider1 = OllamaEmbeddingProvider(
            model="test-model",
            host="http://localhost:11434/",
        )
        provider2 = OllamaEmbeddingProvider(
            model="test-model",
            host="http://localhost:11434",
        )

        assert provider1.host == "http://localhost:11434"
        assert provider2.host == "http://localhost:11434"
        assert provider1.host == provider2.host


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider"""

    @pytest.fixture
    def provider(self):
        """Create an OpenAIEmbeddingProvider instance for testing"""
        return OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="sk-test-key",
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_embed_single_text_success(self, provider):
        """Test successful single text embedding"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], index=0),
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)

        with patch.object(
            provider.client.embeddings, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await provider.embed(["Hello, world!"])

            assert isinstance(result, EmbeddingResponse)
            assert len(result.data) == 1
            assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert result.data[0].index == 0
            assert result.model == "text-embedding-3-small"
            assert result.dimensions == 5

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, provider):
        """Test successful batch embedding"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3], index=0),
            MagicMock(embedding=[0.4, 0.5, 0.6], index=1),
            MagicMock(embedding=[0.7, 0.8, 0.9], index=2),
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = MagicMock(prompt_tokens=15, total_tokens=15)

        with patch.object(
            provider.client.embeddings, "create", new=AsyncMock(return_value=mock_response)
        ):
            texts = ["First", "Second", "Third"]
            result = await provider.embed(texts)

            assert isinstance(result, EmbeddingResponse)
            assert len(result.data) == 3
            assert result.data[0].embedding == [0.1, 0.2, 0.3]
            assert result.data[1].embedding == [0.4, 0.5, 0.6]
            assert result.data[2].embedding == [0.7, 0.8, 0.9]
            assert result.dimensions == 3

    @pytest.mark.asyncio
    async def test_embed_empty_input(self, provider):
        """Test handling of empty input list"""
        result = await provider.embed([])

        assert isinstance(result, EmbeddingResponse)
        assert len(result.data) == 0
        assert result.model == "text-embedding-3-small"
        # Dimensions is 0 when no configured dimensions and empty input
        assert result.dimensions == 0
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_embed_model_override(self, provider):
        """Test that model can be overridden per-call"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2], index=0)]
        mock_response.model = "text-embedding-ada-002"
        mock_response.usage = MagicMock(prompt_tokens=3, total_tokens=3)

        mock_create = AsyncMock(return_value=mock_response)
        with patch.object(provider.client.embeddings, "create", new=mock_create):
            result = await provider.embed(["Test"], model="text-embedding-ada-002")

            # Verify the overridden model was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "text-embedding-ada-002"
            assert result.model == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_usage_tracking(self, provider):
        """Test that usage statistics are tracked"""
        # Check initial state
        assert provider.get_usage() is None

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

        with patch.object(
            provider.client.embeddings, "create", new=AsyncMock(return_value=mock_response)
        ):
            await provider.embed(["Test text"])

            # Verify usage was tracked
            usage = provider.get_usage()
            assert usage is not None
            assert isinstance(usage, Usage)
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 0  # Embeddings don't have completion tokens
            assert usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_dimensions_match_vector_length(self, provider):
        """Test that dimensions field matches actual embedding vector length"""
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=embedding_vector, index=0)]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)

        with patch.object(
            provider.client.embeddings, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await provider.embed(["Test"])

            assert result.dimensions == len(embedding_vector)
            assert result.dimensions == len(result.data[0].embedding)

    @pytest.mark.asyncio
    async def test_dimension_reduction(self):
        """Test that dimensions parameter is passed for dimension reduction"""
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="sk-test-key",
            dimensions=256,  # Request reduced dimensions
        )

        mock_response = MagicMock()
        # Response should have reduced dimensions
        mock_response.data = [MagicMock(embedding=[0.1] * 256, index=0)]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)

        mock_create = AsyncMock(return_value=mock_response)
        with patch.object(provider.client.embeddings, "create", new=mock_create):
            result = await provider.embed(["Test"])

            # Verify dimensions parameter was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["dimensions"] == 256
            assert result.dimensions == 256

    @pytest.mark.asyncio
    async def test_empty_input_with_dimensions(self):
        """Test empty input returns configured dimensions"""
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="sk-test-key",
            dimensions=256,
        )

        result = await provider.embed([])

        assert result.dimensions == 256
        assert len(result.data) == 0


class TestCreateEmbeddingProviderFactory:
    """Tests for create_embedding_provider() factory function"""

    def test_create_ollama_provider(self):
        """Test creating an Ollama embedding provider"""
        config = ModelConfig(
            name="nomic-embed-text",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434",
        )

        provider = create_embedding_provider(config, timeout=60.0)

        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.model == "nomic-embed-text"
        assert provider.timeout == 60.0

    def test_create_ollama_provider_default_url(self):
        """Test creating Ollama provider with default URL"""
        config = ModelConfig(
            name="nomic-embed-text",
            provider=Provider.OLLAMA,
        )

        provider = create_embedding_provider(config)

        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.host == "http://localhost:11434"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
    def test_create_openai_provider(self):
        """Test creating an OpenAI embedding provider"""
        config = ModelConfig(
            name="text-embedding-3-small",
            provider=Provider.OPENAI,
            api_key="sk-test-key",
        )

        provider = create_embedding_provider(config, timeout=30.0)

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model == "text-embedding-3-small"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
    def test_create_openai_provider_with_base_url(self):
        """Test creating OpenAI provider with custom base URL"""
        config = ModelConfig(
            name="text-embedding-3-small",
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            base_url="https://api.openrouter.ai/api/v1",
        )

        provider = create_embedding_provider(config)

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model == "text-embedding-3-small"

    def test_create_provider_unsupported_type(self):
        """Test that unsupported provider raises ValueError"""
        config = ModelConfig(
            name="test-model",
            provider="unsupported",  # type: ignore
        )

        with pytest.raises(ValueError, match="Unsupported provider"):
            create_embedding_provider(config)


class TestEmbeddingResponseEdgeCases:
    """Additional edge case tests for embedding responses"""

    @pytest.mark.asyncio
    async def test_ollama_large_batch(self):
        """Test handling of large batch embedding"""
        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            host="http://localhost:11434",
        )

        # Create mock response with 100 embeddings
        mock_embeddings = [[0.1, 0.2, 0.3]] * 100
        mock_response = {
            "embeddings": mock_embeddings,
            "prompt_eval_count": 500,
        }

        with patch.object(provider.client, "embed", new=AsyncMock(return_value=mock_response)):
            texts = [f"Text {i}" for i in range(100)]
            result = await provider.embed(texts)

            assert len(result.data) == 100
            for i, item in enumerate(result.data):
                assert item.index == i

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
    @pytest.mark.asyncio
    async def test_openai_large_batch(self):
        """Test handling of large batch embedding with OpenAI"""
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )

        # Create mock response with 100 embeddings
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3], index=i) for i in range(100)
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = MagicMock(prompt_tokens=500, total_tokens=500)

        with patch.object(
            provider.client.embeddings, "create", new=AsyncMock(return_value=mock_response)
        ):
            texts = [f"Text {i}" for i in range(100)]
            result = await provider.embed(texts)

            assert len(result.data) == 100
            for i, item in enumerate(result.data):
                assert item.index == i

    def test_embedding_data_with_high_dimensional_vector(self):
        """Test EmbeddingData with a high-dimensional vector"""
        # 1536 dimensions is common for OpenAI text-embedding-ada-002
        embedding = [0.001] * 1536
        data = EmbeddingData(embedding=embedding, index=0)

        assert len(data.embedding) == 1536
        assert data.index == 0

    def test_embedding_response_usage_cumulative_behavior(self):
        """Test that usage reflects only the last call"""
        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            host="http://localhost:11434",
        )

        # Initially no usage
        assert provider.get_usage() is None

        # After a call with usage, it should be set
        with patch.object(
            provider.client,
            "embed",
            new=AsyncMock(
                return_value={"embeddings": [[0.1]], "prompt_eval_count": 10}
            ),
        ):
            import asyncio

            asyncio.get_event_loop().run_until_complete(provider.embed(["Test"]))
            usage1 = provider.get_usage()
            assert usage1 is not None
            assert usage1.prompt_tokens == 10

        # After another call, usage should be updated
        with patch.object(
            provider.client,
            "embed",
            new=AsyncMock(
                return_value={"embeddings": [[0.1]], "prompt_eval_count": 20}
            ),
        ):
            asyncio.get_event_loop().run_until_complete(provider.embed(["Another test"]))
            usage2 = provider.get_usage()
            assert usage2 is not None
            assert usage2.prompt_tokens == 20  # Updated, not cumulative
