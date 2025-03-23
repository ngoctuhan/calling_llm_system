# Call Center Information System

An automated call center system for answering questions using AI, FastAPI, and gRPC.

## Architecture

This system follows a modular architecture with the following components:

- **User Interface**: Interface for user interactions
- **Security Layer**: Authentication, authorization, data encryption, and audit logging
- **AI Interact Layer**: NLP pipeline, context manager, and LLM services
- **Workflow Engine**: Data processor, department mapping, task tracker, and task planner
- **Retrieval Layer**: Text2SQL, knowledge retrieval, full-text search, and cache manager
- **Tools Engine**: Various utility services implemented with gRPC
- **Data Ingestion**: Connectors, loaders, processors, and indexers
- **Data Sources**: External and internal data sources

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env file with your configurations
   ```
5. Start the application:
   ```
   uvicorn app.main:app --reload
   ```

## Project Structure

```
app/
├── api/                 # FastAPI routes
├── core/                # Core application settings
├── db/                  # Database models and connections
├── services/            # AI Interact Layer services
│   ├── nlp_pipeline/    # NLP processing
│   ├── context_manager/ # Context handling
│   └── llm_services/    # LLM integration
├── workflows/           # Workflow Engine components
├── retrieval/           # Retrieval Layer components
├── tools_engine/        # gRPC-based tool services
├── data_ingestion/      # Data ingestion pipeline
├── security/            # Security layer implementation
├── user_interface/      # UI components
└── analytics/           # Analytics and reporting
```

## Development

### Running tests
```
pytest
```

### Starting the gRPC services
```
python app/tools_engine/server.py
``` 