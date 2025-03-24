from typing import Dict, Any, Optional, Union, List, Type
import logging
from enum import Enum

from retrieval_engine.knowledge_retrieval.base_rag import BaseRAG
from retrieval_engine.knowledge_retrieval.vector_rag.vector_rag import VectorRAG
from retrieval_engine.knowledge_retrieval.vector_rag.reranker import RerankMethod
try:
    from retrieval_engine.knowledge_retrieval.graph_rag.graph_rag import GraphRAGQueryEngine
    from retrieval_engine.knowledge_retrieval.graph_rag.graph_builder import GraphRAGBuilder
except ImportError:
    print("Graph RAG llama-index is not installed. It will be disabled.")
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphRAG
from retrieval_engine.knowledge_retrieval.graph_rag_v2.graph_extractor import GraphExtractor
from retrieval_engine.knowledge_retrieval.graph_rag_v2.neo4j_connection import SimpleNeo4jConnection
from llm_services import LLMProviderFactory
from retrieval_engine.knowledge_retrieval.graph_rag_v2.embeddings import create_embedding
from retrieval_engine.knowledge_retrieval.hybrid_rag import HybridRAG

logger = logging.getLogger(__name__)

class RAGType(Enum):
    """Enum for different types of RAG implementations"""
    VECTOR = "vector"
    GRAPH = "graph"
    GRAPH_V2 = "graph_v2"
    HYBRID = "hybrid"

class RetrievalFactory:
    """
    Factory class for creating and configuring different RAG systems.
    
    This class provides a unified interface for creating and configuring
    different types of RAG implementations (Vector-based, Graph-based, etc.)
    with flexible parameter options.
    """
    
    @staticmethod
    def create_rag(
        rag_type: Union[str, RAGType],
        llm_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        retrieval_config: Optional[Dict[str, Any]] = None,
        reranker_config: Optional[Dict[str, Any]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseRAG:
        """
        Create a RAG system with the specified configuration.
        
        Args:
            rag_type (Union[str, RAGType]): Type of RAG system to create
            llm_config (Optional[Dict[str, Any]]): Configuration for the LLM
            embedding_config (Optional[Dict[str, Any]]): Configuration for embeddings
            retrieval_config (Optional[Dict[str, Any]]): Configuration for retrieval
            reranker_config (Optional[Dict[str, Any]]): Configuration for reranking
            hybrid_config (Optional[Dict[str, Any]]): Configuration for hybrid RAG
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseRAG: The configured RAG system
        """
        # Convert string rag_type to enum if needed
        if isinstance(rag_type, str):
            try:
                rag_type = RAGType(rag_type.lower())
            except ValueError:
                valid_types = [t.value for t in RAGType]
                raise ValueError(f"Invalid RAG type: {rag_type}. Valid types are: {valid_types}")
        
        # Initialize configurations with empty dicts if None
        llm_config = llm_config or {}
        embedding_config = embedding_config or {}
        retrieval_config = retrieval_config or {}
        reranker_config = reranker_config or {}
        hybrid_config = hybrid_config or {}
        
        # Create the appropriate RAG system based on type
        if rag_type == RAGType.VECTOR:
            return RetrievalFactory._create_vector_rag(
                llm_config=llm_config,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                reranker_config=reranker_config,
                **kwargs
            )
        elif rag_type == RAGType.GRAPH:
            
            return RetrievalFactory._create_graph_rag(
                llm_config=llm_config,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                **kwargs
            )
        elif rag_type == RAGType.GRAPH_V2:
            return RetrievalFactory._create_graph_rag_v2(
                llm_config=llm_config,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                **kwargs
            )
        elif rag_type == RAGType.HYBRID:
            return RetrievalFactory._create_hybrid_rag(
                llm_config=llm_config,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                reranker_config=reranker_config,
                hybrid_config=hybrid_config,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
    
    @staticmethod
    def _create_vector_rag(
        llm_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        retrieval_config: Dict[str, Any],
        reranker_config: Dict[str, Any],
        **kwargs
    ) -> VectorRAG:
        """
        Create a Vector-based RAG system.
        
        Args:
            llm_config: Configuration for the LLM (not used in VectorRAG directly)
            embedding_config: Configuration for embeddings
            retrieval_config: Configuration for Qdrant vector store
            reranker_config: Configuration for document reranking
            
        Returns:
            VectorRAG: The configured Vector RAG system
        """
        # Extract embedding configuration
        embedding_provider = embedding_config.get("provider", "huggingface")
        embedding_model_name = embedding_config.get("model_name")
        embedding_api_key = embedding_config.get("api_key")
        cache_embeddings = embedding_config.get("cache", True)
        
        # Extract vector store configuration
        collection_name = retrieval_config.get("collection_name", "callcenter")
        qdrant_host = retrieval_config.get("qdrant_host", "localhost")
        qdrant_port = retrieval_config.get("qdrant_port", 6333)
        qdrant_grpc_port = retrieval_config.get("qdrant_grpc_port", 6334)
        qdrant_api_key = retrieval_config.get("qdrant_api_key")
        
        # Extract reranker configuration
        reranker_strategy = reranker_config.get("strategy", RerankMethod.CROSS_ENCODER)
        reranker_model_name = reranker_config.get("model_name")
        
        # Create VectorRAG with configured parameters
        return VectorRAG(
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            embedding_api_key=embedding_api_key,
            reranker_model_name=reranker_model_name,
            reranker_strategy=reranker_strategy,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_grpc_port=qdrant_grpc_port,
            qdrant_api_key=qdrant_api_key,
            cache_embeddings=cache_embeddings,
            **kwargs
        )
    
    @staticmethod
    def _create_graph_rag(
        llm_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        retrieval_config: Dict[str, Any],
        **kwargs
    ) -> GraphRAGQueryEngine:
        """
        Create a Graph-based RAG system v1 - base on llama-index
        
        Args:
            llm_config: Configuration for the LLM
            embedding_config: Configuration for embeddings
            retrieval_config: Configuration for Neo4j graph store
            
        Returns:
            GraphRAGQueryEngine: The configured Graph RAG system
        """
        # Extract LLM configuration
        llm_model_name = llm_config.get("model_name")
        llm_api_key = llm_config.get("api_key")
        llm_provider = llm_config.get("provider", "openai")
        
        # Import the appropriate LLM based on provider
        if llm_provider.lower() == "openai":
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(model=llm_model_name, api_key=llm_api_key)
        elif llm_provider.lower() == "gemini":
            from llama_index.llms.gemini import Gemini
            llm = Gemini(model=llm_model_name, api_key=llm_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # Extract embedding configuration
        embedding_model_name = embedding_config.get("model_name")
        embedding_api_key = embedding_config.get("api_key")
        embedding_provider = embedding_config.get("provider", "openai")
        
        # Set up embedding model
        if embedding_provider.lower() == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding(model=embedding_model_name, api_key=embedding_api_key)
        elif embedding_provider.lower() == "google":
            from llama_index.embeddings.google import GeminiEmbedding
            embed_model = GeminiEmbedding(model=embedding_model_name, api_key=embedding_api_key)
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
            
        # Extract Neo4j configuration
        neo4j_username = retrieval_config.get("username", "neo4j")
        neo4j_password = retrieval_config.get("password", "password")
        neo4j_url = retrieval_config.get("url", "neo4j://127.0.0.1:7687")
        
        # Extract retrieval parameters
        similarity_top_k = retrieval_config.get("similarity_top_k", 20)
        
        # Create graph builder
        graph_builder = GraphRAGBuilder(
            llm=llm,
            username=neo4j_username,
            password=neo4j_password,
            url=neo4j_url,
            embed_model=embed_model,
            **kwargs
        )
        
        # If graph is already built, we should have an index
        index = graph_builder.index
        
        # Create graph query engine
        return GraphRAGQueryEngine(
            model_name=llm_model_name,
            graph_store=graph_builder.graph_store,
            index=index,
            llm=llm,
            similarity_top_k=similarity_top_k
        )
    
    @staticmethod
    def _create_graph_rag_v2(
        llm_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        retrieval_config: Dict[str, Any],
        **kwargs
    ) -> GraphRAG:
        """
        Create a Graph-based RAG system v2 - base on neo4j
        """
        # Extract LLM configuration
        llm_model_name = llm_config.get("model_name")
        llm_api_key = llm_config.get("api_key")
        
        # Extract embedding configuration
        embedding_model_name = embedding_config.get("model_name")
        embedding_api_key = embedding_config.get("api_key")
        embedding_provider = embedding_config.get("provider", "openai")
        
        # Extract Neo4j configuration
        neo4j_username = retrieval_config.get("username", "neo4j")
        neo4j_password = retrieval_config.get("password", "password")
        neo4j_url = retrieval_config.get("url", "neo4j://127.0.0.1:7687")
        
        llm = LLMProviderFactory.create_provider(llm_model_name, 
                                                 llm_api_key)

        embedding_provider = create_embedding(
            provider_type=embedding_provider,
            model_name=embedding_model_name,
            api_key=embedding_api_key,
            cache=False
        )

        # Create graph extractor    
        graph_extractor = GraphExtractor(
            llm=llm,
            embedding_provider=embedding_provider
        )
        
        # Create Neo4j connection
        neo4j_connection = SimpleNeo4jConnection(
            uri=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Create GraphRAG
        return GraphRAG(
            graph_store=neo4j_connection,
            graph_extractor=graph_extractor,
            embedding_provider=embedding_provider,
            **kwargs
        )

    @staticmethod
    def _create_hybrid_rag(
        llm_config: Dict[str, Any],
        embedding_config: Dict[str, Any],
        retrieval_config: Dict[str, Any],
        reranker_config: Dict[str, Any],
        hybrid_config: Dict[str, Any],
        **kwargs
    ) -> HybridRAG:
        """
        Create a Hybrid RAG system that combines Vector and Graph RAG.
        
        Args:
            llm_config: Configuration for the LLM
            embedding_config: Configuration for embeddings
            retrieval_config: Configuration for retrieval stores
            reranker_config: Configuration for reranking
            hybrid_config: Configuration for combining results
            
        Returns:
            HybridRAG: The configured Hybrid RAG system
        """
        # Extract hybrid configuration
        vector_weight = retrieval_config.get("vector_weight", 0.5)
        graph_weight = retrieval_config.get("graph_weight", 0.5)
        combination_strategy = retrieval_config.get("combination_strategy", "weighted")
        deduplicate = retrieval_config.get("deduplicate", True)
        max_workers = retrieval_config.get("max_workers", 2)
        
        vector_rag = retrieval_config.get("vector_rag")
        graph_rag = retrieval_config.get("graph_rag")
        
        # Create the hybrid RAG with both systems
        return HybridRAG(
            vector_rag=vector_rag,
            graph_rag=graph_rag,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            combination_strategy=combination_strategy,
            model_name=embedding_config.get("model_name"),
            deduplicate=deduplicate,
            max_workers=max_workers
        ) 