import os
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from decimal import Decimal

import numpy as np
from retrieval_engine.knowledge_retrieval.vector_rag.qdrant_connection import VectorStore
from retrieval_engine.knowledge_retrieval.vector_rag.embeddings import EmbeddingFactory, EmbeddingProvider

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle Decimal values
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class QueryCache:
    """
    Cache for storing and retrieving natural language queries and their corresponding SQL.
    
    Uses vector similarity search to find semantically similar queries that have been
    asked before, allowing for reuse of previously generated SQL.
    """
    
    def __init__(
        self,
        collection_name: str = "text2sql_queries",
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_provider_type: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        vector_store: Optional[VectorStore] = None,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the query cache
        
        Args:
            collection_name: Name of the Qdrant collection for storing query vectors
            embedding_provider: Custom embedding provider (if None, will be created)
            embedding_provider_type: Type of embedding provider (openai, huggingface, google)
            embedding_model: Name of the embedding model to use
            vector_store: Custom vector store (if None, will be created)
            similarity_threshold: Threshold for similarity search (0-1)
        """
        # Initialize or use provided embedding provider
        if embedding_provider is None:
            logger.info(f"Creating embedding provider of type {embedding_provider_type} with model {embedding_model}")
            self.embedding_provider = EmbeddingFactory.create_provider(
                provider_type=embedding_provider_type,
                model_name=embedding_model,
                cache=True
            )
        else:
            self.embedding_provider = embedding_provider
            
        # Initialize or use provided vector store
        self.vector_store = vector_store or VectorStore()
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        
        # Ensure the collection exists
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """
        Ensure the vector collection exists in Qdrant
        """
        try:
            # Check if collection exists first to avoid recreation
            collections = self.vector_store.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name} for text2sql cache")
                self.vector_store.create_collection(
                    collection_name=self.collection_name,
                    vector_size=self.embedding_provider.dimension
                )
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
    
    def store_query(
        self,
        natural_query: str,
        sql_query: str,
        tables_used: List[str],
        database_name: str,
        execution_success: bool = True,
        execution_result: Optional[Any] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a natural language query and its corresponding SQL in the cache
        
        Args:
            natural_query: The natural language query
            sql_query: The generated SQL query
            tables_used: List of tables referenced in the SQL query
            database_name: Name of the database this query applies to
            execution_success: Whether the SQL query executed successfully
            execution_result: Brief summary of execution result (optional)
            execution_time: Time taken to execute the query in seconds (optional)
            metadata: Additional metadata to store (optional)
            
        Returns:
            bool: True if storing was successful, False otherwise
        """
        try:
            # Generate embedding for the natural query
            query_embedding = self.embedding_provider.embed_query(natural_query)
            
            # Prepare payload
            payload = {
                "natural_query": natural_query,
                "sql_query": sql_query,
                "tables_used": tables_used,
                "database_name": database_name,
                "execution_success": execution_success,
                "timestamp": time.time(),
            }
            
            # Add optional fields if provided
            if execution_result is not None:
                # Convert to string representation if not already a string
                if not isinstance(execution_result, str):
                    if isinstance(execution_result, (list, dict)):
                        # Use custom JSON encoder to handle Decimal values
                        execution_result = json.dumps(execution_result, cls=DecimalEncoder)[:1000]  # Limit size
                    else:
                        execution_result = str(execution_result)[:1000]  # Limit size
                payload["execution_result"] = execution_result
                
            if execution_time is not None:
                payload["execution_time"] = execution_time
                
            if metadata is not None:
                # Convert any Decimal values in metadata to float
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if isinstance(value, Decimal):
                            metadata[key] = float(value)
                payload["metadata"] = metadata
            
            # Store in vector database
            self.vector_store.insert_vectors(
                collection_name=self.collection_name,
                vectors=np.array([query_embedding]),
                payloads=[payload]
            )
            
            logger.info(f"Stored query in cache: {natural_query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error storing query in cache: {str(e)}")
            return False
    
    def find_similar_queries(
        self,
        natural_query: str,
        database_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar natural language queries in the cache
        
        Args:
            natural_query: The natural language query to search for
            database_name: If provided, only return queries for this database
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar queries with their SQL and metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_provider.embed_query(natural_query)
            
            # Prepare filter if database_name is specified
            filter_conditions = None
            if database_name:
                filter_conditions = {
                    "database_name": database_name
                }
            
            # Search for similar queries
            results = self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=self.similarity_threshold,
                filter_conditions=filter_conditions
            )
            
            if results:
                logger.info(f"Found {len(results)} similar queries for: {natural_query[:50]}...")
            else:
                logger.info(f"No similar queries found for: {natural_query[:50]}...")
                
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {str(e)}")
            return []
    
    def get_query_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific query by its ID
        
        Args:
            query_id: The ID of the query to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: The query data if found, None otherwise
        """
        try:
            result = self.vector_store.client.retrieve(
                collection_name=self.collection_name,
                ids=[query_id]
            )
            
            if result and len(result) > 0:
                return result[0].payload
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving query by ID: {str(e)}")
            return None
    
    def clear_cache(self) -> bool:
        """
        Clear all cached queries
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.vector_store.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared query cache collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False 