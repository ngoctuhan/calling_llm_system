# Utils Package

This package contains utilities used across the application, particularly for the chat feature.

## Structure

- `models.py`: Pydantic models used for request/response handling
- `websocket/`: WebSocket handling utilities
  - `connection.py`: WebSocket connection management
- `llm_helpers/`: Helpers for interaction with LLM services
  - `streaming.py`: Utilities for streaming LLM responses
  - `processors.py`: Processing modules for different chat modes

## Chat Modes

The chat system supports multiple modes, each with specialized processing:

1. **Vector Knowledge** (`vector`): 
   - Uses vector embeddings to retrieve semantically similar information
   - Optimized for semantic similarity searches
   - Processes documents through `process_knowledge_mode()`

2. **Graph Knowledge** (`graph`): 
   - Uses graph database (Neo4j) to retrieve knowledge with semantic relationships
   - Returns a minimum of 10 results as requested
   - Includes relationships between nodes for more context
   - Processes through specialized `process_graph_mode()`

3. **Hybrid RAG** (`hybrid_rag`): 
   - Combines vector and graph-based retrieval (60% vector, 40% graph weighting)
   - Deduplicates and combines results from both sources
   - Provides richer context by leveraging both semantic similarity and relationships
   - Processes through specialized `process_hybrid_rag_mode()`

4. **Database** (`database`): 
   - Uses Text-to-SQL to answer questions from structured database
   - Converts natural language to SQL queries
   - Returns both the SQL query and the data
   - Processes through `process_database_mode()`

5. **Hybrid** (`hybrid`): 
   - Combines knowledge base (vector by default) and database for comprehensive answers
   - Gets information from both sources and combines results
   - Useful for questions that need both unstructured and structured data
   - Processes using a combination of knowledge retrieval and database modes

## Processing Flow

Each mode follows these general steps:
1. Extract question and mode from the WebSocket request
2. Initialize the appropriate retrieval systems based on mode
3. Process the question using the specialized processor method for that mode
4. Stream the response back to the client
5. Send the final result with sources, SQL (if applicable), and data (if applicable)

## Refactoring Notes

The original chat.py file was refactored to improve code organization, maintainability, and reusability. The refactored version:

1. Separates concerns with dedicated modules
2. Reduces code duplication
3. Makes the codebase more maintainable and testable
4. Enhances error handling
5. Adds support for multiple retrieval strategies with specialized processors

## Usage

The WebSocket chat endpoint is available at `/api/v1/chat/ws/chat`. To use it, specify the desired mode in the request:

```javascript
socket.send(JSON.stringify({
    question: "Your question",
    mode: "vector" // or "graph", "hybrid_rag", "database", "hybrid"
}));
```

## Future Improvements

Potential future improvements include:
- Adding unit tests for modules
- Implementing more specialized handlers for different response types
- Adding metrics collection
- Implementing caching mechanisms
- Supporting custom weights for hybrid RAG retrieval 