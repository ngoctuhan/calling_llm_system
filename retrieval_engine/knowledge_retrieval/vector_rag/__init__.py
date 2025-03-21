"""
Vector-based RAG package for retrieval augmented generation.

This package provides the components for implementing vector-based retrieval
using Qdrant as the vector database and optional document reranking.
"""

from .vector_rag import VectorRAG
from .reranker import DocumentReranker, RerankMethod
from .qdrant_connection import VectorStore
from .embeddings import (
    EmbeddingProvider,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    GoogleAIEmbeddings,
    CachedEmbeddingProvider,
    EmbeddingFactory
)

__all__ = [
    "VectorRAG",
    "DocumentReranker",
    "RerankMethod",
    "VectorStore",
    "EmbeddingProvider",
    "HuggingFaceEmbeddings",
    "OpenAIEmbeddings",
    "GoogleAIEmbeddings",
    "CachedEmbeddingProvider",
    "EmbeddingFactory"
] 