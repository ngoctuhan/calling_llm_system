# Unified Retrieval Factory

> **COMING SOON: Graph RAG v2.0**  
> A new version of Graph RAG will soon replace the current llama-index based implementation.
> Features include:
> - Native implementation without llama-index dependencies
> - Advanced customization options for graph construction and querying
> - Significant performance optimizations for faster response times
> - Improved memory efficiency and scalability
> - Full compatibility with the existing Retrieval Factory interface

The Unified Retrieval Factory provides a flexible and configurable interface for creating different types of Retrieval Augmented Generation (RAG) systems with customizable parameters.

## Features

- **Multiple RAG Types**: Choose between Vector-based, Graph-based, or Hybrid RAG systems
- **Configurable LLMs**: Use different LLM providers (OpenAI, Google Gemini)
- **Flexible Embeddings**: Configure embedding models from multiple providers (HuggingFace, OpenAI, Google)
- **Reranking Options**: Select different reranking strategies (Cross-Encoder, BM25, MMR)
- **Hybrid Retrieval**: Combine outputs from Vector and Graph RAG for improved results
- **Parallel Retrieval**: Run Vector and Graph retrievals simultaneously for better performance
- **Extensible Design**: Easy to add new RAG implementations or configurations

## Usage

### Basic Usage

```python
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType

# Create a Vector RAG system
vector_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.VECTOR,
    embedding_config={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "api_key": "your-openai-api-key",
    },
    retrieval_config={
        "collection_name": "my_documents",
    },
    reranker_config={
        "strategy": "cross_encoder",
    }
)

# Use the RAG system
results = vector_rag.process(
    query="How do call centers handle customer inquiries?",
    top_k=5,
    rerank=True
)
```

### Creating Vector RAG

Vector RAG systems use vector embeddings and a vector database (Qdrant) for retrieval:

```python
from retrieval_engine.knowledge_retrieval import RetrievalFactory
from retrieval_engine.knowledge_retrieval.vector_rag.reranker import RerankMethod

vector_rag = RetrievalFactory.create_rag(
    rag_type="vector",  # String works too!
    embedding_config={
        "provider": "huggingface",  # Options: "huggingface", "openai", "google"
        "model_name": "all-MiniLM-L6-v2",
        "api_key": None,  # Not needed for HuggingFace
        "cache": True
    },
    retrieval_config={
        "collection_name": "my_collection",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "qdrant_grpc_port": None,
        "qdrant_api_key": None
    },
    reranker_config={
        "strategy": RerankMethod.CROSS_ENCODER,  # Options: CROSS_ENCODER, BM25, MMR, DEFAULT
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Only needed for CROSS_ENCODER
    }
)
```

### Creating Graph RAG

Graph RAG systems use knowledge graphs and Neo4j for retrieval:

```python
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType

graph_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.GRAPH,
    llm_config={
        "provider": "openai",  # Options: "openai", "gemini"
        "model_name": "gpt-4",
        "api_key": "your-openai-api-key"
    },
    embedding_config={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "api_key": "your-openai-api-key"
    },
    retrieval_config={
        "username": "neo4j",
        "password": "password",
        "url": "neo4j://localhost:7687",
        "similarity_top_k": 20
    }
)
```

### Create Graph RAG v2 

Graph RAG v2, it is update from version 1 about performances and workflow

```python
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType

graph_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.GRAPH_V2,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    embedding_config={
        "provider": "google",  
        "model_name": "text-embedding-004",  
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    retrieval_config={
        "username": "neo4j",
        "password": "password",
        "url": "neo4j://localhost:7687",
        "similarity_top_k": 10
    }
)
```
### Creating Hybrid RAG

Hybrid RAG combines Vector and Graph RAG approaches for improved retrieval:

```python
from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType

# Tạo Graph RAG v2
graph_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.GRAPH_V2,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    embedding_config={
        "provider": "google",  
        "model_name": "text-embedding-004",  
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    retrieval_config={
        "username": "neo4j",
        "password": "password",
        "url": "neo4j://localhost:7687",
        "similarity_top_k": 10
    }
)

# Tạo Vector RAG
vector_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.VECTOR,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    embedding_config={
        "provider": "huggingface",  
        "model_name": "all-MiniLM-L6-v2",  
    },
    retrieval_config={
        "index_name": "example_hf", 
        "similarity_top_k": 10
    }
)

# Tạo Hybrid RAG
hybrid_rag = RetrievalFactory.create_rag(
    rag_type=RAGType.HYBRID,
    llm_config={
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_API_KEY")
    },
    retrieval_config={
        "vector_rag": vector_rag,
        "graph_rag": graph_rag,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "combination_strategy": "weighted",  # 'weighted', 'ensemble', hoặc 'cascade'
        "deduplicate": True,
        "max_workers": 2
    }
)

print("\n=== Kết quả từ Graph RAG ===")
graph_results = graph_rag.process("Nguyễn Trãi mất năm nào?", top_k=5)
print(graph_results)

print("\n=== Kết quả từ Vector RAG ===")
vector_results = vector_rag.process("Nguyễn Trãi mất năm nào?", top_k=5)
print(vector_results)

print("\n=== Kết quả từ Hybrid RAG ===")
hybrid_results = hybrid_rag.process("Nguyễn Trãi mất năm nào?", top_k=5)
print(hybrid_results)

# Đóng kết nối
import asyncio
asyncio.run(hybrid_rag.close())

```

## Configuration Options

### Common Configuration

| Parameter | Description |
|-----------|-------------|
| `rag_type` | Type of RAG system to create (vector, graph, hybrid) |

### LLM Configuration

| Parameter | Description |
|-----------|-------------|
| `provider` | LLM provider (openai, gemini) |
| `model_name` | Name of the LLM model |
| `api_key` | API key for the LLM provider |

### Embedding Configuration

| Parameter | Description |
|-----------|-------------|
| `provider` | Embedding provider (huggingface, openai, google) |
| `model_name` | Name of the embedding model |
| `api_key` | API key for the embedding provider |
| `cache` | Whether to cache embeddings (default: True) |

### Vector Retrieval Configuration

| Parameter | Description |
|-----------|-------------|
| `collection_name` | Name of the Qdrant collection |
| `qdrant_host` | Qdrant server host |
| `qdrant_port` | Qdrant server port |
| `qdrant_grpc_port` | Qdrant server gRPC port |
| `qdrant_api_key` | Qdrant API key |

### Graph Retrieval Configuration

| Parameter | Description |
|-----------|-------------|
| `username` | Neo4j username |
| `password` | Neo4j password |
| `url` | Neo4j connection URL |
| `similarity_top_k` | Number of top entities to retrieve |

### Reranker Configuration

| Parameter | Description |
|-----------|-------------|
| `strategy` | Reranking strategy (cross_encoder, bm25, mmr, default) |
| `model_name` | Reranker model name (for cross_encoder) |

### Hybrid Configuration

| Parameter | Description |
|-----------|-------------|
| `vector_weight` | Weight for vector results (0.0 to 1.0) |
| `graph_weight` | Weight for graph results (0.0 to 1.0) |
| `combination_strategy` | How to combine results (weighted, ensemble, cascade) |
| `deduplicate` | Whether to remove duplicate results |
| `max_workers` | Maximum number of parallel workers for retrieval (default: 2) |
| `vector_config` | Specific configuration for the Vector RAG component |
| `graph_config` | Specific configuration for the Graph RAG component |
| `use_vector` | Whether to use Vector RAG (default: True) |
| `use_graph` | Whether to use Graph RAG (default: True) |

## Hybrid Combination Strategies

The Hybrid RAG supports different strategies for combining results from Vector and Graph approaches:

1. **Weighted (default)**: Combines results based on assigned weights, normalizing scores across both systems
2. **Ensemble**: Interleaves results from both systems, alternating between vector and graph results
3. **Cascade**: Tries the vector approach first, and falls back to graph results if vector results are insufficient

## Performance Optimization

The Hybrid RAG system includes several performance optimizations:

1. **Parallel Retrieval**: Vector and Graph retrievals run simultaneously using ThreadPoolExecutor
2. **Configurable Parallelism**: Control the number of worker threads with the `max_workers` parameter
3. **Automatic Fallback**: Falls back to sequential execution when only one RAG system is available
4. **Error Isolation**: Errors in one retrieval system don't affect the other system's results

## Extension

To add support for a new RAG type:

1. Create a new implementation that inherits from `BaseRAG`
2. Add a new entry in the `RAGType` enum
3. Add a new private method in `RetrievalFactory` to create the new RAG type
4. Update the `create_rag` method to handle the new type

Example:

```python
# Add to RAGType enum
class RAGType(Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    NEW_TYPE = "new_type"  # Add new type here

# Add new factory method
@staticmethod
def _create_new_type_rag(
    llm_config: Dict[str, Any],
    embedding_config: Dict[str, Any],
    retrieval_config: Dict[str, Any],
    **kwargs
) -> NewTypeRAG:
    # Implementation here
    pass

# Update create_rag method
if rag_type == RAGType.NEW_TYPE:
    return RetrievalFactory._create_new_type_rag(
        llm_config=llm_config,
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
        **kwargs
    ) 