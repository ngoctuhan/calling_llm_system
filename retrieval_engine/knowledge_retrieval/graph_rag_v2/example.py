#!/usr/bin/env python3
"""
Example usage of Graph RAG v2 with Neo4j
"""

import os
import asyncio
import logging
from typing import Dict, Any

from graph_rag_v2 import GraphRAG, create_embedding

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def simple_example():
    """Simple example using OpenAI embeddings with a small document"""
    # Create GraphRAG with default configurations
    graph_rag = GraphRAG(
        # Neo4j configuration (from environment variables)
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "password"),
        
        # Use OpenAI for both extraction and embedding
        graph_extractor_type="openai",
        graph_extractor_model="gpt-4o",
        embedding_provider_type="openai",
        embedding_model="text-embedding-3-small",
        
        # Enable caching for better performance
        enable_caching=True
    )
    
    # Document with AI info
    sample_doc = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to human intelligence.
    Machine learning is a subset of AI that enables systems to learn from data rather than through explicit programming.
    Deep learning is a subset of machine learning using neural networks with multiple layers.
    Natural language processing (NLP) is a branch of AI that focuses on the interaction between computers and humans using natural language.
    Computer vision is an AI field that enables computers to derive meaningful information from digital images and videos.
    """
    
    # Process document
    await graph_rag.initialize()
    result = await graph_rag.process_document(text=sample_doc, document_id="ai_concepts")
    print(f"Processing result: {result}")
    
    # Query the knowledge graph
    query = "What is the relationship between AI and machine learning?"
    results = await graph_rag.query(query_text=query)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Subject: {result.get('subject')}, Predicate: {result.get('predicate')}, Object: {result.get('object')}")
    
    # Close connection
    await graph_rag.close()

async def advanced_example():
    """Advanced example using HuggingFace embeddings with multiple documents"""
    # Create GraphRAG with customized configurations
    graph_rag = GraphRAG(
        # Neo4j configuration
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "password"),
        
        # Use OpenAI for extraction but HuggingFace for embeddings
        graph_extractor_type="openai",
        graph_extractor_model="gpt-4o",
        embedding_provider_type="huggingface",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,  # Specify the dimension for the model
        
        # Process configuration
        max_workers=4,
        default_chunk_size=500,
        default_chunk_overlap=50
    )
    
    # Documents about different programming languages
    documents = [
        {
            "text": """Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
            Python emphasizes code readability with its notable use of significant whitespace.
            Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.
            Python has a large and comprehensive standard library.""",
            "document_id": "python_info"
        },
        {
            "text": """JavaScript is a high-level, often just-in-time compiled language that conforms to the ECMAScript specification.
            JavaScript has curly-bracket syntax, dynamic typing, prototype-based object-orientation, and first-class functions.
            JavaScript is one of the core technologies of the World Wide Web, enabling interactive web pages.
            JavaScript engines were originally used only in web browsers, but they are now core components of other software systems.""",
            "document_id": "javascript_info"
        }
    ]
    
    # Process documents
    await graph_rag.initialize()
    results = await graph_rag.process_documents(documents=documents)
    print(f"Processing results: {results}")
    
    # Query
    query = "What programming paradigms does Python support?"
    results = await graph_rag.query(query_text=query)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Subject: {result.get('subject')}, Predicate: {result.get('predicate')}, Object: {result.get('object')}")
    
    # Close connection
    await graph_rag.close()

async def embedding_comparison_example():
    """Example demonstrating different embedding providers from vector_rag"""
    
    # Sample text for comparison
    sample_text = "Graph RAG uses knowledge graphs to store and retrieve information."
    
    # Initialize different embedding providers
    openai_embedder = create_embedding(service_type="openai", model_name="text-embedding-3-small")
    
    huggingface_embedder = create_embedding(service_type="huggingface", model_name="all-MiniLM-L6-v2")
    
    # Only initialize Google if API key is available
    google_embedder = None
    if os.environ.get("GOOGLE_API_KEY"):
        google_embedder = create_embedding(service_type="google", model_name="embedding-001")
    
    # Generate embeddings
    openai_embedding = openai_embedder.embed_query(sample_text)
    huggingface_embedding = huggingface_embedder.embed_query(sample_text)
    
    # Print information
    print("\nEmbedding Comparison:")
    print(f"OpenAI: model={openai_embedder.model_name}, dimension={openai_embedder.dimension}")
    print(f"HuggingFace: model={huggingface_embedder.model_name}, dimension={huggingface_embedder.dimension}")
    
    if google_embedder:
        google_embedding = google_embedder.embed_query(sample_text)
        print(f"Google: model={google_embedder.model_name}, dimension={google_embedder.dimension}")

async def component_example():
    """Example showing how to use components individually with direct vector_rag embeddings"""
    
    from graph_rag_v2 import (
        create_graph_extractor,
        create_embedding,
        Neo4jConnection
    )
    
    # Initialize components
    extractor = create_graph_extractor(
        extractor_type="openai",
        model="gpt-4o"
    )
    
    # Using direct vector_rag embeddings
    embedder = create_embedding(
        provider_type="openai",
        model_name="text-embedding-3-small",
        cache=True
    )
    
    neo4j = Neo4jConnection(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password")
    )
    
    # Sample text
    text = "The Graph RAG architecture combines knowledge graphs with Large Language Models for better retrieval."
    
    # Extract triplets
    await neo4j.connect()
    triplets = await extractor.extract_triplets(text=text)
    
    print("\nExtracted Triplets:")
    for triplet in triplets:
        print(f"Subject: {triplet.subject}, Predicate: {triplet.predicate}, Object: {triplet.object}")
    
    # Get embeddings
    entities = [triplet.subject for triplet in triplets] + [triplet.object for triplet in triplets]
    entity_embeddings = {entity: embedder.embed_query(entity) for entity in entities}
    
    print(f"\nGenerated embeddings for {len(entity_embeddings)} entities")
    
    # Close connection
    await neo4j.close()

if __name__ == "__main__":
    """Run examples"""
    
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set in environment variables")
    
    if not os.environ.get("NEO4J_URI"):
        print("Warning: NEO4J_URI not set, using default: bolt://localhost:7687")
    
    # Run examples
    asyncio.run(simple_example())
    # asyncio.run(advanced_example())
    # asyncio.run(embedding_comparison_example())
    # asyncio.run(component_example()) 