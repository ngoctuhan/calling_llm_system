#!/usr/bin/env python3
"""
GraphRAG Example Script

This example demonstrates how to use GraphRAG to query knowledge from a Neo4j graph,
using semantic search and structure-based graph search.
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
import traceback
# Load environment variables from .env file
load_dotenv()

# Add root directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import required modules
    from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphRAG
    from retrieval_engine.knowledge_retrieval.graph_rag_v2.graph_extractor import GraphExtractor
    from retrieval_engine.knowledge_retrieval.graph_rag_v2.neo4j_connection import SimpleNeo4jConnection
    from llm_services import LLMProviderFactory
    from retrieval_engine.knowledge_retrieval.graph_rag_v2.embeddings import create_embedding
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please check the path and install dependencies")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of sample queries
SAMPLE_QUERIES = [
    # English queries
    "Who was Albert Einstein?",
    "What did Einstein discover?",
    "When did Einstein receive the Nobel Prize?",
    
    # Queries for Vietnamese content
    "Who was Nguyen Trai?",
    "What uprising did Nguyen Trai participate in?",
    "What is Hanoi famous for?",
    
    # Complex queries
    "Relationship between Einstein and quantum theory",
    "Describe the scientific contributions of Einstein",
    "Relationship between Nguyen Trai and the Later Le Dynasty"
]

async def get_graph_rag():
    # Initialize embedding provider
    embedding_provider = create_embedding(
        provider_type="google",
        model_name="text-embedding-004",
        api_key=os.getenv("GOOGLE_API_KEY"),
        cache=False
    )

    # Initialize LLM provider
    llm = LLMProviderFactory.create_provider(
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    graph_extractor = GraphExtractor(llm=llm, 
                                     embedding_provider=embedding_provider)
    
    neo4j_connection = SimpleNeo4jConnection(
        uri=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password")
    )

    return GraphRAG(
        graph_store=neo4j_connection,
        graph_extractor=graph_extractor,
        embedding_provider=embedding_provider,
        similarity_threshold=0.4
    )

async def test_semantic_search():
    """Test semantic search based on vector embeddings"""
    logger.info("=== Starting semantic search test ===")
    
    # Initialize GraphRAG
    rag = await get_graph_rag()
    
    # Select a sample query
    query = "Who was Albert Einstein and what did he discover?"
    
    try:
        # Perform semantic search
        results = rag.retrieve(
            query=query,
            top_k=5,
            use_semantic=True,
            use_graph=False
        )
        
        # Print results
        logger.info(f"Semantic search results for query: '{query}'")
        logger.info(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['text']}")
            logger.info(f"   Confidence: {result['metadata']['confidence']}")
            logger.info(f"   Source: {result['metadata']['source']}")
            logger.info("---")
        
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
    
    logger.info("=== Completed semantic search test ===")

async def test_graph_search():
    """Test structure-based graph search"""
    logger.info("=== Starting graph search test ===")
    
    # Initialize GraphRAG
    rag = await get_graph_rag()
    
    # Select a sample query
    query = "What uprising did Nguyen Trai participate in?"
    
    try:
        # Perform graph search
        results = rag.retrieve(
            query=query,
            top_k=5,
            use_semantic=False,
            use_graph=True
        )
        
        # Print results
        logger.info(f"Graph search results for query: '{query}'")
        logger.info(f"Found {len(results)} results:")
        
        print("results: ", results)
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['text']}")
            if 'triplet' in result:
                logger.info(f"   Subject: {result['triplet']['subject']}")
                logger.info(f"   Predicate: {result['triplet']['predicate']}")
                logger.info(f"   Object: {result['triplet']['object']}")
            logger.info("---")
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error performing graph search: {e}")
    
    logger.info("=== Completed graph search test ===")

async def test_hybrid_search():
    """Test hybrid search combining semantic and graph-based approaches"""
    logger.info("=== Starting hybrid search test ===")
    
    # Initialize GraphRAG
    rag = await get_graph_rag()
    
    # Select a sample query
    query = "Describe scientific contributions and awards of Einstein"
    
    try:
        # Perform hybrid search
        results = rag.retrieve(
            query=query,
            top_k=10,
            use_semantic=True,
            use_graph=True
        )
        
        # Print results
        logger.info(f"Hybrid search results for query: '{query}'")
        logger.info(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['text']}")
            logger.info(f"   Confidence: {result['metadata'].get('confidence', 'N/A')}")
            logger.info(f"   Source: {result['metadata'].get('source', 'N/A')}")
            if 'seed_entity' in result['metadata'] and result['metadata']['seed_entity']:
                logger.info(f"   Seed Entity: {result['metadata']['seed_entity']}")
            logger.info("---")
        
        # Perform reranking
        reranked = rag.rerank(query, results, top_k=5)
        
        logger.info(f"Results after reranking (top 5):")
        for i, result in enumerate(reranked, 1):
            logger.info(f"{i}. {result['text']}")
            logger.info(f"   Confidence: {result['metadata'].get('confidence', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error performing hybrid search: {e}")
    
    logger.info("=== Completed hybrid search test ===")

async def test_query_processing():
    """Test complete query processing through BaseRAG process"""
    logger.info("=== Starting query processing test ===")
    
    # Initialize GraphRAG with reranker model
    rag = await get_graph_rag()
    
    # Select a sample query
    query = "What are the major contributions of Albert Einstein to physics?"
    
    try:
        # Perform full query processing (retrieve + rerank)
        results = rag.process(
            query=query,
            top_k=10,
            rerank=True,
            rerank_top_k=5
        )
        
        # Print results
        logger.info(f"Query processing results for: '{query}'")
        logger.info(f"Metadata: {results['metadata']}")
        logger.info(f"Reranked: {results['reranked']}")
        logger.info(f"Found {len(results['documents'])} results:")
        
        for i, doc in enumerate(results['documents'], 1):
            logger.info(f"{i}. {doc['text']}")
            logger.info(f"   Confidence: {doc['metadata'].get('confidence', 'N/A')}")
        
        # Format context for LLM
        context = rag.format_context(results, include_metadata=True)
        logger.info(f"Context formatted for LLM:\n{context[:500]}...")
        
    except Exception as e:
        logger.error(f"Error performing query processing: {e}")
    
    logger.info("=== Completed query processing test ===")

async def main():
    """Main function to execute the program"""
    logger.info("Starting GraphRAG example")
    
    # Check required settings
    required_envs = ["GOOGLE_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_envs = [env for env in required_envs if not os.environ.get(env)]
    
    if missing_envs:
        logger.error(f"Missing environment variables: {', '.join(missing_envs)}")
        logger.info("Please set the following environment variables:")
        logger.info("export GOOGLE_API_KEY=<your_google_key>")
        logger.info("export NEO4J_URI=neo4j://localhost:7687")
        logger.info("export NEO4J_USERNAME=neo4j")
        logger.info("export NEO4J_PASSWORD=<your_password>")
        return
    
    # Run tests
    # await test_semantic_search()
    logger.info("\n" + "="*50 + "\n")
    
    await test_graph_search()
    logger.info("\n" + "="*50 + "\n")
    
    # await test_hybrid_search()
    # logger.info("\n" + "="*50 + "\n")
    
    # await test_query_processing()
    
    logger.info("Completed GraphRAG example")

if __name__ == "__main__":
    asyncio.run(main()) 