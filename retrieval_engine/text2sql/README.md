# Text2SQL System

A system for converting natural language questions into SQL queries. The system connects to a database, analyzes the schema, and generates optimized SQL based on the user's natural language question.

## Features

- **Database Connectivity**: Connect to PostgreSQL databases (with MySQL support planned)
- **Query Caching**: Store and retrieve previously executed queries using vector similarity search
- **Schema Analysis**: Identify relevant tables based on the query and schema
- **SQL Generation**: Generate optimized SQL for the target database
- **Fuzzy Verification**: Verify and correct SQL syntax and schema references
- **Error Handling**: Automatic retry with error feedback

## Installation

```bash
# Install dependencies
pip install psycopg2-binary openai python-dotenv qdrant-client
```

## Environment Variables

Create a `.env` file with the following variables:

```
# PostgreSQL configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password

# OpenAI API configuration
OPENAI_API_KEY=your_api_key

# Qdrant vector database configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Usage

```python
from retrieval_engine.text2sql import Text2SQL

# Initialize the system
text2sql = Text2SQL(db_type="postgres")

# Process a natural language query
result = text2sql.process_query("What are the top 5 products by sales?")

# Get the SQL and results
if result["success"]:
    print(f"Generated SQL: {result['sql']}")
    print(f"Results: {result['data']}")
else:
    print(f"Error: {result['error']}")
```

## Architecture

The Text2SQL system consists of several components:

1. **DBConnector**: Abstract base class with PostgreSQL implementation to connect to the database and fetch schema information
2. **QueryCache**: Caches natural language queries and their SQL equivalents for faster retrieval
3. **Text2SQL**: Main class that orchestrates the conversion process

### Processing Flow

1. **Retrieval**: Check if a similar query has been processed before
2. **Schema Analysis**: Identify relevant tables based on query keywords
3. **SQL Generation**: Generate SQL query using LLM with schema context
4. **Verification**: Check SQL for syntax errors and schema correctness
5. **Execution**: Run the query and handle any execution errors
6. **Caching**: Store successful queries for future use

## Limitations

- Currently only supports SELECT queries for security reasons
- Limited to PostgreSQL databases
- Requires OpenAI API key for SQL generation

## Future Improvements

- Support for additional database types (MySQL, SQLite, etc.)
- Improved table selection algorithm
- Support for more complex query types
- Integration with additional LLM providers 