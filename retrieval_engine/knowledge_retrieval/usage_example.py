"""
Usage examples for the flexible Retrieval Factory.

This file demonstrates various ways to configure and use the RAG systems
through the unified RetrievalFactory interface.
"""

from retrieval_engine.knowledge_retrieval import RetrievalFactory, RAGType
from retrieval_engine.knowledge_retrieval.vector_rag.reranker import RerankMethod


def vector_rag_example():
    """Example of creating and using a Vector-based RAG"""
    
    # Create Vector RAG with OpenAI embeddings and cross-encoder reranking
    vector_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.VECTOR,
        embedding_config={
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "your-openai-api-key",
            "cache": True
        },
        retrieval_config={
            "collection_name": "my_documents",
            "qdrant_host": "localhost",
            "qdrant_port": 6333
        },
        reranker_config={
            "strategy": RerankMethod.CROSS_ENCODER,
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
    )
    
    # Add documents to the vector store
    documents = [
        "Call centers often handle customer service inquiries via phone and chat.",
        "Vector databases store embeddings for efficient similarity search.",
        "LLMs have transformed natural language processing capabilities."
    ]
    
    metadatas = [
        {"source": "call_center_doc", "category": "customer_service"},
        {"source": "vector_db_doc", "category": "databases"},
        {"source": "llm_doc", "category": "machine_learning"}
    ]
    
    vector_rag.index_documents(documents=documents, metadatas=metadatas)
    
    # Retrieve information with default process method
    results = vector_rag.process(
        query="How do call centers work?",
        top_k=3,
        rerank=True,
        filter_conditions={"category": "customer_service"}
    )
    
    print("Vector RAG Results:")
    for i, doc in enumerate(results["documents"], 1):
        print(f"Result {i}: {doc['content']} (Score: {doc.get('score', 'N/A')})")
    
    return vector_rag


def graph_rag_example():
    """Example of creating and using a Graph-based RAG"""
    
    # Create Graph RAG with Gemini LLM and OpenAI embeddings
    graph_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.GRAPH,
        llm_config={
            "provider": "gemini",
            "model_name": "models/gemini-1.5-pro",
            "api_key": "your-google-api-key"
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
    
    # Example query (assuming graph already has data)
    results = graph_rag.retrieve(query="What is the relationship between Facebook and WhatsApp?")
    
    print("\nGraph RAG Results:")
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result}")
    
    return graph_rag


def hybrid_rag_example():
    """Example of creating and using a Hybrid RAG that combines Vector and Graph RAG"""
    
    # Create Hybrid RAG with both Vector and Graph capabilities
    hybrid_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.HYBRID,
        # Shared LLM configuration
        llm_config={
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": "your-openai-api-key"
        },
        # Shared embedding configuration
        embedding_config={
            "provider": "openai",
            "model_name": "text-embedding-3-small",
            "api_key": "your-openai-api-key"
        },
        # Main retrieval config (will be shared and overridden as needed)
        retrieval_config={},
        # Reranker configuration for Vector RAG
        reranker_config={
            "strategy": RerankMethod.CROSS_ENCODER,
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        # Hybrid-specific configuration
        hybrid_config={
            # Weights for combining results
            "vector_weight": 0.6,
            "graph_weight": 0.4,
            # Combination strategy: weighted, ensemble, or cascade
            "combination_strategy": "weighted",
            # Whether to deduplicate combined results
            "deduplicate": True,
            # Vector-specific configuration
            "vector_config": {
                "retrieval_config": {
                    "collection_name": "hybrid_documents",
                    "qdrant_host": "localhost",
                    "qdrant_port": 6333
                }
            },
            # Graph-specific configuration
            "graph_config": {
                "retrieval_config": {
                    "username": "neo4j",
                    "password": "password",
                    "url": "neo4j://localhost:7687",
                    "similarity_top_k": 20
                }
            }
        }
    )
    
    # Retrieve information with combined approaches
    results = hybrid_rag.retrieve(
        query="How do Facebook and WhatsApp handle user privacy?",
        top_k=5,
        # You can pass specific arguments to each RAG type
        vector_kwargs={
            "filter_conditions": {"category": "privacy"}
        },
        graph_kwargs={
            # Graph-specific parameters
        }
    )
    
    print("\nHybrid RAG Results:")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}: {doc['content']} (Score: {doc.get('score', 'N/A')}, Source: {doc['metadata'].get('source_type', 'unknown')})")
    
    # Process query with reranking
    processed_results = hybrid_rag.process(
        query="What privacy features does WhatsApp offer?",
        top_k=5,
        rerank=True
    )
    
    print("\nHybrid RAG Processed Results (with reranking):")
    for i, doc in enumerate(processed_results["documents"], 1):
        print(f"Result {i}: {doc['content']} (Score: {doc.get('score', 'N/A')}, Source: {doc['metadata'].get('source_type', 'unknown')})")
    
    return hybrid_rag


def flexible_rag_creation():
    """Demonstrate flexible RAG creation with different configurations"""
    
    # 1. Create Vector RAG with HuggingFace embeddings and MMR reranking
    huggingface_vector_rag = RetrievalFactory.create_rag(
        # Can use string instead of enum
        rag_type="vector",
        embedding_config={
            "provider": "huggingface",
            "model_name": "all-MiniLM-L6-v2"
        },
        reranker_config={
            "strategy": RerankMethod.MMR,
            # No model needed for MMR
        },
        retrieval_config={
            "collection_name": "huggingface_collection"
        }
    )
    
    # 2. Create Vector RAG with Google AI embeddings and BM25 reranking
    google_vector_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.VECTOR,
        embedding_config={
            "provider": "google",
            "model_name": "text-embedding-004",
            "api_key": "your-google-api-key"
        },
        reranker_config={
            "strategy": RerankMethod.BM25
        },
        retrieval_config={
            "collection_name": "google_collection"
        }
    )
    
    # 3. Create Graph RAG with OpenAI LLM
    openai_graph_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.GRAPH,
        llm_config={
            "provider": "openai",
            "model_name": "gpt-4-turbo",
            "api_key": "your-openai-api-key"
        },
        retrieval_config={
            "username": "neo4j",
            "password": "neo4j_password"
        }
    )
    
    # 4. Create Hybrid RAG with cascade strategy
    cascade_hybrid_rag = RetrievalFactory.create_rag(
        rag_type=RAGType.HYBRID,
        llm_config={
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": "your-openai-api-key"
        },
        embedding_config={
            "provider": "openai",
            "model_name": "text-embedding-3-small"
        },
        hybrid_config={
            "combination_strategy": "cascade",  # Try vector first, fall back to graph
            "vector_config": {
                "retrieval_config": {"collection_name": "cascade_collection"}
            }
        }
    )
    
    print("Successfully created multiple RAG configurations")
    return huggingface_vector_rag, google_vector_rag, openai_graph_rag, cascade_hybrid_rag


if __name__ == "__main__":
    print("Running Vector RAG example...")
    vector_rag = vector_rag_example()
    
    print("\nRunning Graph RAG example...")
    graph_rag = graph_rag_example()
    
    print("\nRunning Hybrid RAG example...")
    hybrid_rag = hybrid_rag_example()
    
    print("\nDemonstrating flexible RAG creation...")
    flexible_rag_creation() 