import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import math
from functools import lru_cache

from ..base_rag import BaseRAG
from .neo4j_connection import SimpleNeo4jConnection
from .graph_extractor import GraphExtractor
from .embeddings import EmbeddingProvider
from retrieval_engine.knowledge_retrieval.vector_rag.reranker import DocumentReranker, RerankMethod

logger = logging.getLogger(__name__)

class GraphRAG(BaseRAG):
    """
    Graph-based Retrieval Augmented Generation (RAG) system
    
    This implementation uses a knowledge graph stored in Neo4j to provide 
    context-aware retrieval through semantic search.
    """
    
    def __init__(
        self,
        # Neo4j configuration
        graph_store: SimpleNeo4jConnection,
        graph_extractor: GraphExtractor,
        embedding_provider: EmbeddingProvider,
        
        # Query configuration
        top_k_entities: int = 5,
        similarity_threshold: float = 0.7,
        max_results: int = 20,
        
        # Reranker configuration
        reranker_strategy: Union[str, RerankMethod] = RerankMethod.CROSS_ENCODER,
        reranker_model_name: Optional[str] = None,
    ):
        """
        Initialize GraphRAG
        
        Args:
            graph_store: Neo4j connection
            graph_extractor: Graph extraction utility
            embedding_provider: Embedding provider
            top_k_entities: Number of top entities to use as seeds
            similarity_threshold: Threshold for semantic similarity
            max_results: Maximum number of results to return
            reranker_strategy: Strategy for reranking (cross_encoder, bm25, mmr, default)
            reranker_model_name: Name of the reranker model
        """
        
        self._neo4j = graph_store
        self._extractor = graph_extractor
        self._embedder = embedding_provider
        
        # Query settings
        self.top_k_entities = top_k_entities
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Initialize reranker
        self.reranker = DocumentReranker(
            strategy=reranker_strategy,
            model_name=reranker_model_name
        )
    
    async def close(self):
        """Close all connections"""
        if self._neo4j:
            self._neo4j.close()
        
        self._initialized = False
        logger.info("GraphRAG connections closed")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge graph information based on a query using only semantic search.
        
        Args:
            query (str): The user query
            top_k (int): Number of results to retrieve
            **kwargs: Additional parameters:
                similarity_threshold: Threshold for semantic similarity
            
        Returns:
            List[Dict[str, Any]]: List of retrieved triplets
        """  
        try:
            # Generate embedding for semantic search
            query_embedding = self._embedder.embed_query(query)
            
            # Get similarity threshold from kwargs or default
            similarity_threshold = kwargs.get("similarity_threshold", self.similarity_threshold)
            
            # Run the async query
            results = asyncio.run(self._retrieve_semantic(
                query_text=query,
                query_embedding=query_embedding,
                top_k_entities=self.top_k_entities,
                limit=top_k * 2,  # Get more results to filter
                similarity_threshold=similarity_threshold
            ))
            
            # Format the results to match the BaseRAG expected format
            formatted_results = []
            for result in results:
                # Convert triplet to text format - include description if available
                description = result.get('description', '')
                if description:
                    text = description
                else:
                    text = f"{result['subject']} {result['predicate']} {result['object']}"
                
                formatted_result = {
                    "text": text,
                    "triplet": f"""({result.get("subject")}) --[{result.get("predicate")}]--> ({result.get("object")})""",
                    "metadata": {
                        "source": result.get("source", "knowledge_graph"),
                        "document_id": result.get("document_id", "")
                    }
                }
                formatted_results.append(formatted_result)
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return []
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using the configured reranker.
        
        Args:
            query (str): The original user query
            documents (List[Dict]): List of retrieved documents to rerank
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Dict[str, Any]]: Reranked documents with scores
        """
        try:
            # Format documents for reranker
            formatted_docs = []
            for doc in documents:
                formatted_doc = {
                    "id": doc.get("metadata", {}).get("document_id", ""),
                    "content": doc.get("text", ""),
                    "score": 1.0,  # Default score
                    "metadata": doc.get("metadata", {})
                }
                formatted_docs.append(formatted_doc)
            
            # Rerank documents
            reranked_docs = self.reranker.rerank(
                query=query,
                documents=formatted_docs,
                top_k=top_k
            )
            
            # Format results back to original structure
            results = []
            for doc in reranked_docs:
                result = {
                    "text": doc.get("content", ""),
                    "triplet": doc.get("metadata", {}).get("triplet", ""),
                    "metadata": {
                        "source": doc.get("metadata", {}).get("source", "knowledge_graph"),
                        "document_id": doc.get("id", ""),
                        "score": doc.get("score", 1.0)
                    }
                }
                results.append(result)
            
            logger.info(f"Reranked {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            # If reranking fails, return the original documents
            return documents[:top_k]
    
    async def _retrieve_semantic(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k_entities: int = 5,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve triplets using semantic search
        
        1. Find top k nearest entities
        2. Get all triplets where these entities appear
        3. Return triplets with descriptions
        
        Args:
            query_text: The query text
            query_embedding: The query embedding vector
            top_k_entities: Number of top entities to use as seeds
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching triplets
        """
        # Find top k entities by semantic similarity
        similar_entities = await self._find_similar_entities(
            query_embedding=query_embedding,
            limit=top_k_entities,
            similarity_threshold=similarity_threshold
        )
        
        if not similar_entities:
            logger.info("No similar entities found for the query")
            return []
        
        # Extract entity names and scores
        entity_data = [(entity["entity"], entity["score"]) for entity in similar_entities]
        entity_names = [entity for entity, _ in entity_data]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Found {len(entity_names)} similar entities: {entity_names}")
        else:
            logger.info(f"Found {len(entity_names)} similar entities")

        # Get all triplets involving these entities
        triplets = await self._get_entity_triplets(
            entity_names=entity_names,
            entity_scores=dict(entity_data),
            limit=limit
        )
        
        logger.info(f"Retrieved {len(triplets)} triplets for the query")
        return triplets
    
    async def _find_similar_entities(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Find similar entities to the query embedding
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Number of entities to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar entities and their similarity scores
        """
        # Perform vector search through Neo4j for entities
        result = self._neo4j.run_vector_search(
            query_embedding=query_embedding,
            node_type="entity",
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        # Format the results to have a consistent structure
        formatted_results = []
        for item in result:
            formatted_results.append({
                "entity": item.get("name", ""),
                "score": item.get("score", 0.0)
            })
            
        return formatted_results
    
    async def _get_entity_triplets(
        self, 
        entity_names: List[str], 
        entity_scores: Dict[str, float],
        limit: int = 20
    ):
        """
        Get all triplets where the specified entities appear as either subject or object
        
        Args:
            entity_names: List of entity names to find triplets for
            entity_scores: Dictionary mapping entity names to similarity scores
            limit: Maximum number of results
            
        Returns:
            List of triplets with descriptions and context
        """
        # Optimized single query that handles both subject and object cases
        unified_query = """
        MATCH (seed:Entity)-[r1]-(mid:Entity)
        WHERE seed.name IN $entity_names OR mid.name IN $entity_names
        AND type(r1) <> 'FROM_DOCUMENT'
        RETURN DISTINCT
            seed.name AS subject,
            type(r1) AS predicate_type,
            r1.name AS predicate,
            r1.description AS description,
            mid.name AS object,
            CASE 
                WHEN seed.name IN $entity_names THEN seed.name
                ELSE mid.name
            END AS seed_entity
        ORDER BY seed_entity
        """
        
        params = {
            "entity_names": entity_names,
        }
        
        try:
            # Execute the unified query
            results = self._neo4j.execute_query(unified_query, params)
            # Use dictionary to keep only one result per description
            unique_results = {}
            
            for record in results:
                description = record.get("description", "")
                if description not in unique_results:
                    unique_results[description] = {
                        "subject": record["subject"],
                        "predicate": record.get("predicate") or record.get("predicate_type"),
                        "object": record["object"],
                        "description": description,
                        "seed_entity": record["seed_entity"],
                        "source": "semantic"
                    }
            # Convert dictionary values to list and limit results
            formatted_results = list(unique_results.values())
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error executing triplet query: {str(e)}")
            return []
    
    