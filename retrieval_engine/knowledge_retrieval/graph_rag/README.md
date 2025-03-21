# Graph RAG System

A powerful and flexible Graph-based Retrieval Augmented Generation (RAG) system that supports:

- Knowledge graph extraction from text using LLMs
- Community detection and summarization
- Entity-based retrieval
- Integration with Neo4j graph database
- Advanced query processing

## System Architecture

The Graph RAG system consists of several key components:

1. **GraphRAGQueryEngine**: The main class that coordinates the graph-based retrieval
2. **GraphRAGExtractor**: Extracts knowledge graph triplets from text
3. **GraphRAGStore**: Manages the graph database and community detection
4. **GraphRAGBuilder**: Orchestrates the graph building process

### Component Overview

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  Knowledge     │     │  Neo4j         │     │  Community     │
│  Extraction    │◄────┤  Graph Store   │◄────┤  Detection     │
│                │     │                │     │                │
└────────┬───────┘     └────────────────┘     └────────┬───────┘
         │                                             │
         │             ┌────────────────┐              │
         │             │                │              │
         └───────────►│  GraphRAG       │◄─────────────┘
                       │  Query Engine  │
                       └────────────────┘
```

## Knowledge Extraction Framework

The knowledge extraction framework provides a flexible way to extract knowledge graphs from text:

- Uses an LLM to identify entities and relationships
- Extracts triplets in the form of "entity1->relation->entity2"
- Processes entities with their types and descriptions
- Handles batch processing of documents

### Extraction Architecture

The extraction system follows a modular design pattern:

1. `GraphRAGExtractor`: Core component that uses an LLM to extract triplets from text
2. Knowledge graph prompt template for guiding the LLM extraction
3. Parsing functions to convert LLM outputs into structured graph data
4. Integration with LlamaIndex for efficient document processing

## Neo4j Graph Store

The `GraphRAGStore` class provides a flexible interface for working with Neo4j:

- **Graph Storage**: Stores entities and relationships in Neo4j
- **Community Detection**: Uses the hierarchical Leiden algorithm to detect communities
- **Community Summarization**: Generates summaries for each detected community
- **Entity-Community Mapping**: Maps entities to their respective communities

## Graph-Based Retrieval

The system includes multiple retrieval methods to improve query results:

- **Entity Detection**: Identifies entities in the user query
- **Community Retrieval**: Finds communities related to the detected entities
- **Summary Retrieval**: Returns community summaries for relevant information

## Usage Examples

### Basic Usage

```python
from retrieval_engine.knowledge_retrieval.graph_rag import GraphRAGBuilder, GraphRAGQueryEngine
from llama_index.llms.gemini import Gemini
from llama_index.core.schema import TextNode

# Initialize LLM
llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key="your-api-key"
)

# Initialize GraphRAGBuilder
graph_builder = GraphRAGBuilder(
    llm=llm,
    username="neo4j", 
    password="password", 
    url="neo4j://127.0.0.1:7687"
)

# Create text nodes
nodes = [
    TextNode(text="Facebook is a social media platform created by Mark Zuckerberg in 2004."),
    TextNode(text="Mark Zuckerberg is the CEO of Meta, which owns Facebook, Instagram, and WhatsApp.")
]

# Build graph
index = graph_builder.build_graph(nodes)

# Build communities
graph_builder.build_communities()

# Initialize query engine
query_engine = GraphRAGQueryEngine(
    model_name=None,
    graph_store=graph_builder.graph_store,
    index=index,
    llm=llm
)

# Retrieve information
results = query_engine.retrieve("Who created Facebook?")
```

### Adding New Nodes

```python
# Add more nodes to the existing graph
new_nodes = [
    TextNode(text="WhatsApp is a messaging app acquired by Facebook in 2014.")
]

# Update the graph with new nodes
graph_builder.add_nodes(new_nodes)

# Rebuild communities to incorporate new information
graph_builder.build_communities()
```

### Retrieving Community Information

```python
# Get all triplets from the graph
triplets = graph_builder.get_triplets()

# Get community summaries
community_summaries = graph_builder.graph_store.get_community_summaries()

# Print community information
for community_id, summary in community_summaries.items():
    print(f"Community {community_id}: {summary}")
```

## Components in Detail

### GraphRAGBuilder

The GraphRAGBuilder orchestrates the process of:
- Initializing the graph extraction process
- Setting up the Neo4j graph store
- Building the knowledge graph from text nodes
- Managing the community building process

### GraphRAGExtractor

The GraphRAGExtractor handles:
- Extracting entity and relationship triplets from text
- Processing text with specialized LLM prompts
- Parsing LLM outputs into structured data
- Handling asynchronous extraction for efficiency

### GraphRAGStore

The GraphRAGStore manages:
- Storage of entities and relationships in Neo4j
- Community detection using network analysis algorithms
- Generation of summaries for each community
- Maintenance of entity-community mappings

### GraphRAGQueryEngine

The GraphRAGQueryEngine provides:
- Entity detection in user queries
- Retrieval of relevant communities
- Access to community summaries for question answering
- Integration with the LlamaIndex framework

## Requirements

To use the Graph RAG system, you'll need:

- **Core**: `pip install llama-index networkx graspologic neo4j`
- **LLM Integration**: `pip install llama-index-llms-openai llama-index-llms-gemini`
- **Embedding Models**: `pip install llama-index-embeddings-openai`
- **Neo4j Database**: Running Neo4j instance (local or cloud)

For all dependencies:
```
pip install llama-index networkx graspologic neo4j llama-index-llms-openai llama-index-llms-gemini llama-index-embeddings-openai
``` 