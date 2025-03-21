#!/usr/bin/env python3
"""
Vector RAG System Complete Example

This example demonstrates the usage of the VectorRAG system with different
embedding providers, reranking methods, and search capabilities.
"""

import os
import logging
import traceback
from pprint import pprint
from retrieval_engine.knowledge_retrieval.vector_rag import (
    VectorRAG, 
    RerankMethod,
    DocumentReranker,
    EmbeddingFactory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Example documents about call centers and related technologies
DOCUMENTS = [
    "Call centers are facilities used by companies to handle large volumes of customer calls.",
    "Vector databases store and retrieve vector embeddings efficiently for similarity search.",
    "RAG systems combine retrieval from a knowledge base with generative AI models.",
    "Embedding models convert text into numerical representations for semantic search.",
    "Qdrant is a vector database optimized for similarity search operations.",
    "Machine learning models are trained on large datasets to make predictions.",
    "Natural Language Processing (NLP) enables computers to understand and process human language.",
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "APIs (Application Programming Interfaces) allow different software systems to communicate.",
    "Data mining involves extracting patterns and insights from large datasets.",
    "Call centers use AI-powered chatbots to handle simple customer queries.",
    "AI systems can analyze customer sentiment during calls.",
    "Call center agents use knowledge bases to answer customer questions quickly.",
    "Speech recognition technologies transcribe call center conversations automatically.",
    "Call routing systems direct customers to the most appropriate agent or department.",
    "Natural language processing helps understand customer inquiries.",
    "Machine learning algorithms predict common customer issues based on historical data.",
    "Customer relationship management (CRM) systems store and organize customer data.",
    "Interactive voice response (IVR) systems automate initial customer interactions.",
    "Call center analytics track key performance metrics like average handling time.",
    "Workforce management systems optimize agent scheduling based on call volume predictions.",
    "Quality assurance in call centers involves monitoring calls for compliance and service quality.",
    "Omnichannel support allows customers to switch between communication channels seamlessly.",
    "Knowledge management systems organize information for quick retrieval by agents.",
    "Call center KPIs include first call resolution, average speed of answer, and abandonment rate.",
    "Cloud-based call centers allow agents to work remotely from any location.",
    "Virtual agents use AI to handle customer interactions without human intervention.",
    "Predictive analytics in call centers forecast call volumes and customer behavior.",
    "Customer satisfaction (CSAT) surveys measure how well agents resolve issues.",
    "Call center automation reduces manual work and improves efficiency."
]

# Metadata for each document
METADATAS = [
    {"category": "call_center_basics" if i < 10 else "call_center_technology", 
     "doc_id": f"doc{i+1}", 
     "source": "knowledge_base",
     "page": i+1} 
    for i in range(len(DOCUMENTS))
]

def example_huggingface():
    """Example using HuggingFace embeddings with basic retrieval and reranking"""
    print("\n\n=== EXAMPLE 1: HuggingFace Embeddings ===")
    
    # Initialize VectorRAG with HuggingFace embeddings
    rag = VectorRAG(
        collection_name="example_hf",
        embedding_provider="huggingface",
        embedding_model_name="all-MiniLM-L6-v2",
        reranker_strategy=RerankMethod.CROSS_ENCODER
    )
    
    # Index documents
    rag.index_documents(documents=DOCUMENTS, metadatas=METADATAS)
    
    # Basic retrieval
    query = "How do call centers use AI technology?"
    print(f"\nQuery: {query}")
    
    results = rag.retrieve(query=query, top_k=5)
    print("\nTop 5 Retrieved Results:")
    for i, doc in enumerate(results):
        print(f"{i+1}. [{doc['score']:.4f}] {doc['content'][:100]}...")
    
    # Reranking
    reranked_results = rag.rerank(query=query, documents=results, top_k=3)
    print("\nTop 3 Reranked Results:")
    for i, doc in enumerate(reranked_results):
        print(f"{i+1}. [New: {doc['score']:.4f}, Original: {doc['original_score']:.4f}] {doc['content'][:100]}...")
    
    return rag

def example_openai():
    """Example using OpenAI embeddings with filtered search"""
    # Skip if no OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\n\n=== EXAMPLE 2: OpenAI Embeddings (SKIPPED - No API key) ===")
        print("To run this example, set the OPENAI_API_KEY environment variable.")
        return None
    
    print("\n\n=== EXAMPLE 2: OpenAI Embeddings with Filtered Search ===")
    
    # Initialize VectorRAG with OpenAI embeddings
    rag = VectorRAG(
        collection_name="example_openai",
        embedding_provider="openai",
        embedding_model_name="text-embedding-3-small",
        reranker_strategy=RerankMethod.DEFAULT
    )
    
    # Index documents
    rag.index_documents(documents=DOCUMENTS, metadatas=METADATAS)
    
    # Filtered search
    query = "AI technology in customer service"
    filter_conditions = {"category": "call_center_technology"}
    
    print(f"\nQuery: {query}")
    print(f"Filter: {filter_conditions}")
    
    results = rag.retrieve(
        query=query, 
        top_k=3, 
        filter_conditions=filter_conditions
    )
    
    print("\nFiltered Search Results (only 'call_center_technology' category):")
    for i, doc in enumerate(results):
        print(f"{i+1}. [{doc['score']:.4f}] {doc['content'][:100]}...")
        print(f"   Metadata: {doc['metadata']}")
    
    return rag

def example_rerankers(rag):
    """Example demonstrating different reranking methods"""
    if not rag:
        print("\n\n=== EXAMPLE 3: Different Reranking Methods (SKIPPED) ===")
        print("Need a working VectorRAG instance from previous examples")
        return
    
    print("\n\n=== EXAMPLE 3: Different Reranking Methods ===")
    
    query = "customer service automation technologies"
    
    # Retrieve initial results
    initial_results = rag.retrieve(query=query, top_k=10)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(initial_results)} initial results")
    
    # Try different reranking strategies
    reranking_methods = [
        RerankMethod.CROSS_ENCODER,
        RerankMethod.BM25,
        RerankMethod.DEFAULT
    ]
    
    # Try MMR if sentence-transformers is available
    try:
        from sentence_transformers import SentenceTransformer
        reranking_methods.append(RerankMethod.MMR)
    except ImportError:
        print("\nMMR reranking requires sentence-transformers (not available)")
    
    for method in reranking_methods:
        print(f"\n--- Reranking with {method.value} ---")
        
        reranker = DocumentReranker(strategy=method)
        reranked = reranker.rerank(
            query=query, 
            documents=initial_results, 
            top_k=5,
            diversity_weight=0.3 if method == RerankMethod.MMR else 0.0
        )
        
        for i, doc in enumerate(reranked):
            print(f"{i+1}. [{doc['score']:.4f}] {doc['content'][:100]}...")

def example_hybrid_search(rag):
    """Example demonstrating hybrid search"""
    if not rag:
        print("\n\n=== EXAMPLE 4: Hybrid Search (SKIPPED) ===")
        print("Need a working VectorRAG instance from previous examples")
        return
    
    print("\n\n=== EXAMPLE 4: Hybrid Search ===")
    
    query = "call center predictive analytics"
    print(f"\nQuery: {query}")
    
    # Hybrid search
    hybrid_results = rag.hybrid_search(
        query=query,
        top_k=5,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    print("\nHybrid Search Results (70% vector, 30% keyword):")
    for i, doc in enumerate(hybrid_results):
        print(f"{i+1}. [{doc['score']:.4f}] {doc['content'][:100]}...")
        if 'vector_score' in doc and 'keyword_score' in doc:
            print(f"   Vector score: {doc['vector_score']:.4f}, Keyword score: {doc['keyword_score']:.4f}")

def example_direct_embedding_usage():
    """Example using embedding providers directly"""
    print("\n\n=== EXAMPLE 5: Direct Embedding Providers Usage ===")
    
    # Create embedding provider through factory
    provider = EmbeddingFactory.create_provider(
        provider_type="huggingface",
        model_name="all-MiniLM-L6-v2",
        cache=True
    )
    
    # Generate embeddings
    query = "How do call centers use AI?"
    query_embedding = provider.embed_query(query)
    
    documents = [
        "Call centers use AI for customer service automation.",
        "AI technologies help predict customer needs in call centers."
    ]
    doc_embeddings = provider.embed_documents(documents)
    
    print(f"\nQuery: {query}")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"Document embedding count: {len(doc_embeddings)}")
    print(f"Document embedding dimension: {len(doc_embeddings[0])}")
    
    # Calculate similarity (dot product)
    import numpy as np
    similarities = [
        np.dot(query_embedding, doc_emb) / 
        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]
    
    print("\nSimilarities:")
    for i, (doc, sim) in enumerate(zip(documents, similarities)):
        print(f"{i+1}. [{sim:.4f}] {doc}")

def run_examples():
    """Run all examples"""
    try:
        # Example 1: HuggingFace embeddings
        rag = example_huggingface()
        
        # Example 2: OpenAI embeddings with filtered search
        openai_rag = example_openai()
        
        # Use whichever RAG instance worked
        working_rag = rag or openai_rag
        
        # Example 3: Different reranking methods
        example_rerankers(working_rag)
        
        # Example 4: Hybrid search
        example_hybrid_search(working_rag)
        
        # Example 5: Direct embedding usage
        example_direct_embedding_usage()
        
        print("\n\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"\n\nError in examples: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_examples() 