"""
Embeddings for Graph RAG v2.
Direct reuse of vector_rag embeddings implementation.
"""

# Import all embedding components directly from vector_rag
from ..vector_rag.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    GoogleAIEmbeddings,
    EmbeddingFactory,
    CachedEmbeddingProvider
)

# Re-export the create_provider function with a more intuitive name for our module
def create_embedding(
    provider_type: str = "openai",
    model_name: str = None,
    api_key: str = None,
    cache: bool = True,
    cache_dir: str = ".embedding_cache",
    **kwargs
):
    """
    Create an embedding provider (direct pass-through to vector_rag's EmbeddingFactory)
    
    Args:
        provider_type: Type of provider ("openai", "huggingface", or "google")
        model_name: Model name for the provider
        api_key: API key for the provider
        cache: Whether to use caching
        cache_dir: Directory for caching embeddings
        **kwargs: Additional provider-specific arguments
        
    Returns:
        The requested embedding provider
    """
    return EmbeddingFactory.create_provider(
        provider_type=provider_type,
        model_name=model_name,
        api_key=api_key,
        cache=cache,
        cache_dir=cache_dir,
        **kwargs
    ) 