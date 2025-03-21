# Graph RAG v2.0

Knowledge graph-based Retrieval Augmented Generation using Neo4j.

## Features

- **Knowledge Graph Extraction**: Extract knowledge triplets (subject-predicate-object) from text using LLMs (OpenAI or Gemini)
- **Async Neo4j Integration**: Store and query knowledge graph in Neo4j database with asynchronous operations
- **Direct Embeddings**: Directly uses vector_rag embeddings without additional wrapper layers
- **Hybrid Search**: Combine semantic similarity with graph structure for comprehensive querying
- **Batch Processing**: Process and index documents in batches with concurrent operations
- **Chunking Strategy**: Intelligent document chunking with customizable sizing and overlap
- **Caching**: Optional embedding caching to reduce API calls and improve performance
- **Modular Design**: Combine components independently or use the full RAG system

## Installation

1. Set up a Neo4j database (local or AuraDB)
2. Install the package dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export OPENAI_API_KEY="your-openai-key"
# Optional for Gemini
export GOOGLE_API_KEY="your-google-key"
```

## Usage

### Simple Example

```python
import asyncio
from graph_rag_v2 import GraphRAG

async def main():
    # Initialize GraphRAG with defaults
    graph_rag = GraphRAG(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        graph_extractor_type="openai",
        embedding_provider_type="openai"
    )
    
    # Process a document
    await graph_rag.initialize()
    await graph_rag.process_document(
        text="Artificial intelligence (AI) is intelligence demonstrated by machines.",
        document_id="ai_intro"
    )
    
    # Query the knowledge graph
    results = await graph_rag.query("What is artificial intelligence?")
    
    # Print results
    for result in results:
        print(f"{result.get('subject')} {result.get('predicate')} {result.get('object')}")
    
    # Close connections
    await graph_rag.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Configuration

GraphRAG supports extensive customization:

```python
graph_rag = GraphRAG(
    # Neo4j configuration
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    
    # Knowledge extraction
    graph_extractor_type="openai",
    graph_extractor_model="gpt-4o",
    graph_extractor_temperature=0.1,
    
    # Embedding providers (direct use of vector_rag embeddings)
    embedding_provider_type="huggingface",
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    enable_caching=True,
    cache_dir=".embedding_cache",
    
    # Processing parameters
    max_workers=4,
    default_chunk_size=500,
    default_chunk_overlap=50,
    
    # Query parameters
    semantic_search_limit=10,
    graph_search_limit=10,
    hybrid_search=True,
    similarity_threshold=0.7
)
```

## Embedding Providers

GraphRAG v2.0 directly uses vector_rag embeddings for consistency. Supported providers:

- **OpenAI**: Fast, high-quality embeddings using OpenAI's API
  - Default model: `text-embedding-3-small`

- **HuggingFace**: Local embedding models with sentence-transformers
  - Default model: `all-MiniLM-L6-v2`

- **Google**: Embeddings from Google's Generative AI models
  - Requires: `GOOGLE_API_KEY`
  - Default model: `embedding-001`

## Components

GraphRAG consists of these main components:

1. **Graph Extractor**: Extracts knowledge triplets from text using LLMs
2. **Embeddings**: Vector embeddings directly from vector_rag
3. **Neo4j Connection**: Manages the connection and operations with the Neo4j database
4. **GraphRAG**: The main class that orchestrates the entire system

## License

MIT
