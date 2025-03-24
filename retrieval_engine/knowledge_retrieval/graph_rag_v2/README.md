# Graph RAG v2

Graph RAG v2 is a Retrieval Augmented Generation (RAG) framework based on knowledge graphs, designed to combine the semantic search capabilities of embedding models with the relationship representation power of knowledge graphs.

## Module Structure

The module is organized into the following main components:

1. **GraphRAG**: The main class that allows querying and searching for information based on graphs, inheriting from BaseRAG.
2. **GraphBuilder**: Class that handles building knowledge graphs from documents.
3. **SimpleNeo4jConnection**: Class that handles connections and queries to the Neo4j database.
4. **GraphExtractor**: Class that extracts information from text into knowledge triplets.

## Functionality

### 1. GraphRAG

Focuses on retrieving information from knowledge graphs:

- Inherits from BaseRAG for compatibility with standard RAG methods
- Supports semantic search based on vector embeddings
- Supports searching based on graph structure
- Combines results from both methods to provide richer results

### 2. GraphBuilder

Focuses on building and maintaining knowledge graphs:

- Processes documents to extract knowledge triplets
- Creates embeddings for entities
- Stores information in Neo4j database
- Supports processing multiple documents simultaneously

**Note**: GraphBuilder does not handle document chunking but processes input text directly. If you need to split long texts, do so before passing them to GraphBuilder or use a separate chunking module.

### 3. SimpleNeo4jConnection

Manages connections and queries to Neo4j:

- Establishes and maintains connections to Neo4j database
- Performs basic and complex queries
- Manages vector embeddings of entities
- Handles storage of knowledge triplets

### 4. GraphExtractor

Extracts information from text:

- Uses LLM to extract knowledge triplets from text
- Converts LLM output into KnowledgeTriplet objects
- Supports batch processing of texts

## Usage Examples

### Creating a Knowledge Graph

```python
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphBuilder

# Initialize GraphBuilder
builder = GraphBuilder(
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    graph_extractor_model="gpt-4o"
)

# Process document
async def process():
    await builder.initialize()
    result = await builder.process_document(
        text="Albert Einstein was a German physicist who developed the theory of relativity.",
        document_id="doc1",
        document_metadata={"source": "Wikipedia"}
    )
    print(f"Processed document: {result}")
    await builder.close()

# Run processing
import asyncio
asyncio.run(process())
```

### Querying the Knowledge Graph

```python
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphRAG

# Initialize GraphRAG
rag = GraphRAG(
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

# Query the graph
results = rag.retrieve(
    query="What did Einstein develop?",
    top_k=5,
    use_semantic=True
)

# Display results
for result in results:
    print(f"- {result['text']}")
```

## Processing Long Documents

For long documents, you can use a separate chunking module or preprocess before passing to GraphBuilder:

```python
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphBuilder
from text_processing import TextChunker  # Separate text chunking module

# Initialize
builder = GraphBuilder(
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

async def process_long_document(text, document_id, metadata=None):
    # Split text into chunks
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split_text(text)
    
    # Process each chunk
    results = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{i}"
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["chunk_index"] = i
        
        # Process chunk with GraphBuilder
        result = await builder.process_document(
            text=chunk,
            document_id=chunk_id,
            document_metadata=chunk_metadata
        )
        results.append(result)
    
    return results
```

## API Documentation

See detailed method descriptions in the docstrings of each class.

## Requirements

- Python 3.7+
- Neo4j database (can use Neo4j Desktop or Neo4j Aura Cloud)
- Python libraries: neo4j, openai (or other supported embedding library)

## Installation Notes

To use vector search in Neo4j, you need to use Neo4j 5.0+ for Vietnamese language processing scenarios.
