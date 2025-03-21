# Vector RAG System

A flexible and extensible Vector-based Retrieval Augmented Generation (RAG) system that supports:

- Multiple embedding providers (HuggingFace, OpenAI, Google AI)
- Advanced document reranking strategies
- Vector storage with Qdrant
- Hybrid search capabilities (vector + keyword)

## System Architecture

The Vector RAG system consists of several key components:

1. **VectorRAG**: The main class that coordinates the RAG pipeline
2. **Embedding Providers**: Convert text to vector embeddings
3. **Reranker**: Improves search relevance with various reranking strategies
4. **Qdrant Connection**: Manages interactions with the vector database

### Component Overview

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  Embedding     │     │  Qdrant        │     │  Document      │
│  Providers     │◄────┤  Vector Store  │◄────┤  Reranker      │
│                │     │                │     │                │
└────────┬───────┘     └────────────────┘     └────────┬───────┘
         │                                             │
         │             ┌────────────────┐              │
         │             │                │              │
         └───────────►│    VectorRAG    │◄─────────────┘
                       │                │
                       └────────────────┘
```

## Embedding Framework

The embeddings framework provides a flexible way to generate vector embeddings from text using multiple providers:

- **HuggingFace** (local models via SentenceTransformers)
- **OpenAI** (API-based embeddings)
- **Google AI** (API-based embeddings)

### Embedding Architecture

The embeddings system follows a modular design pattern:

1. `EmbeddingProvider` abstract base class that defines the interface for all embedding providers
2. Concrete implementations for different providers:
   - `HuggingFaceEmbeddings`: Local embedding models using SentenceTransformers
   - `OpenAIEmbeddings`: OpenAI API-based embeddings
   - `GoogleAIEmbeddings`: Google AI API-based embeddings
3. `CachedEmbeddingProvider`: A wrapper that adds caching functionality to any embedding provider
4. `EmbeddingFactory`: A factory class to create and configure embedding providers easily

## Qdrant Vector Store

The `VectorStore` class provides a flexible interface for working with Qdrant:

- **Unified Search API**: A single `search` method that supports both basic and filtered searches
- **Batch Search**: Efficient batch search capabilities for multiple queries
- **Helper Methods**: Internal methods for creating filters and other common operations
- **Flexible Document Format**: Compatible with nested document structures in Qdrant's payload format

## Reranking Capabilities

The system supports multiple reranking methods to improve retrieval quality:

- **Cross-Encoder**: Uses a cross-encoder model for precise relevance scoring
- **BM25**: Traditional lexical similarity algorithm
- **MMR** (Maximum Marginal Relevance): Balances relevance with diversity
- **Default**: Uses the original vector similarity scores

The `RerankMethod` enum provides a type-safe way to select reranking strategies. The reranker supports both flat document structures and nested Qdrant payload structures.

## Usage Examples

### Basic Usage

```python
from retrieval_engine.knowledge_retrieval.vector_rag import VectorRAG, RerankMethod

# Initialize VectorRAG
rag = VectorRAG(
    collection_name="my_documents",
    embedding_provider="huggingface",
    embedding_model_name="all-MiniLM-L6-v2",
    reranker_strategy=RerankMethod.CROSS_ENCODER
)

# Index documents
documents = [
    "Call centers are facilities used by companies to handle large volumes of customer calls.",
    "Vector databases store and retrieve vector embeddings efficiently for similarity search."
]
metadatas = [
    {"source": "call_center_doc", "category": "customer_service"},
    {"source": "database_doc", "category": "technology"}
]
rag.index_documents(documents=documents, metadatas=metadatas)

# Retrieve documents
results = rag.retrieve(query="How do call centers work?", top_k=3)

# Rerank for better relevance
reranked_results = rag.rerank(query="How do call centers work?", documents=results, top_k=2)
```

### Using Different Embedding Providers

```python
# With OpenAI embeddings
openai_rag = VectorRAG(
    collection_name="openai_collection",
    embedding_provider="openai",
    embedding_model_name="text-embedding-3-small"
)

# With Google AI embeddings
google_rag = VectorRAG(
    collection_name="google_collection",
    embedding_provider="google",
    embedding_model_name="text-embedding-004"
)
```

### Filtered Search

```python
# Retrieve documents with filter conditions
filtered_results = rag.retrieve(
    query="How do call centers work?", 
    top_k=5,
    filter_conditions={"category": "customer_service"}
)
```

### Hybrid Search

```python
# Combine vector search with keyword matching
hybrid_results = rag.hybrid_search(
    query="call center technology",
    top_k=5,
    vector_weight=0.7,  # Weight for vector search scores
    keyword_weight=0.3  # Weight for keyword search scores
)
```

### Creating Embedding Providers Directly

```python
from retrieval_engine.knowledge_retrieval.vector_rag import EmbeddingFactory

# Create a HuggingFace embedding provider
hf_embeddings = EmbeddingFactory.create_provider(
    provider_type="huggingface",
    model_name="all-MiniLM-L6-v2",
    cache=True
)

# Create an OpenAI embedding provider
openai_embeddings = EmbeddingFactory.create_provider(
    provider_type="openai",
    model_name="text-embedding-3-small",
    api_key="your-api-key",  # Optional, can also use OPENAI_API_KEY env var
    cache=True
)

# Create a Google AI embedding provider
google_embeddings = EmbeddingFactory.create_provider(
    provider_type="google",
    model_name="text-embedding-004",
    api_key="your-api-key",  # Optional, can also use GOOGLE_API_KEY env var
    cache=True
)
```

### Working with Reranker Directly

```python
from retrieval_engine.knowledge_retrieval.vector_rag import DocumentReranker, RerankMethod

# Initialize a reranker with cross-encoder strategy
reranker = DocumentReranker(strategy=RerankMethod.CROSS_ENCODER)

# Rerank documents
query = "How do call centers use AI?"
reranked_docs = reranker.rerank(
    query=query,
    documents=retrieved_documents,
    top_k=5,
    diversity_weight=0.3  # Only used for MMR strategy
)

# Evaluate reranker performance
eval_results = reranker.evaluate(
    queries=["How do call centers use AI?", "What technologies are used in call centers?"],
    all_documents=[document_set1, document_set2],
    relevant_ids=[["doc1", "doc2"], ["doc3", "doc4"]],
    metrics=["precision", "recall", "ndcg"]
)
```

## Environment Variables

The system supports the following environment variables for configuration:

- `OPENAI_API_KEY`: API key for OpenAI
- `GOOGLE_API_KEY`: API key for Google AI
- `EMBEDDING_MODEL`: Default model name for HuggingFace
- `OPENAI_EMBEDDING_MODEL`: Default model name for OpenAI
- `GOOGLE_EMBEDDING_MODEL`: Default model name for Google AI
- `RERANKER_MODEL`: Default model for cross-encoder reranking
- `QDRANT_HOST`: Host for Qdrant database (default: localhost)
- `QDRANT_PORT`: Port for Qdrant database (default: 6333)

## Requirements

Different components have different dependencies:

- **Core**: `pip install qdrant-client numpy`
- **HuggingFace embeddings**: `pip install sentence-transformers`
- **OpenAI embeddings**: `pip install openai`
- **Google AI embeddings**: `pip install google-generativeai`
- **BM25 reranking**: `pip install rank_bm25`

For all dependencies:
```
pip install qdrant-client numpy sentence-transformers openai google-genai rank_bm25
``` 