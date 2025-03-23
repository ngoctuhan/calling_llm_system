import os
import logging
import asyncio
from typing import List, Dict, Any, Optional

from ..base_rag import BaseRAG
from .neo4j_connection import SimpleNeo4jConnection
from .graph_extractor import GraphExtractor
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

class GraphRAG(BaseRAG):
    """
    Graph-based Retrieval Augmented Generation (RAG) system
    
    This implementation uses a knowledge graph stored in Neo4j to provide 
    context-aware retrieval through semantic and structured search.
    """
    
    def __init__(
        self,
        # Neo4j configuration
        graph_store: SimpleNeo4jConnection,
        graph_extractor: GraphExtractor,
        embedding_provider: EmbeddingProvider,
        
        # Query configuration
        semantic_search_limit: int = 10,
        graph_search_limit: int = 10,
        hybrid_search: bool = True,
        similarity_threshold: float = 0.7,
        max_hops: int = 2,
        
    ):
        """
        Initialize GraphRAG
        
        Args:
            semantic_search_limit: Maximum number of results from semantic search
            graph_search_limit: Maximum number of results from graph search
            hybrid_search: Whether to use hybrid search (semantic + graph)
            similarity_threshold: Threshold for semantic similarity
            max_hops: Maximum number of hops for graph traversal
        """
        
        self._neo4j = graph_store
        self._extractor = graph_extractor
        self._embedder = embedding_provider
        
        # Query settings
        self.semantic_search_limit = semantic_search_limit
        self.graph_search_limit = graph_search_limit
        self.hybrid_search = hybrid_search
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops
    
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
        Retrieve knowledge graph information based on a query.
        
        Args:
            query (str): The user query
            top_k (int): Number of results to retrieve
            **kwargs: Additional parameters:
                use_semantic: Whether to use semantic search (default: True)
                use_graph: Whether to use graph-based search (default: True if use_semantic is False)
                similarity_threshold: Threshold for semantic similarity
            
        Returns:
            List[Dict[str, Any]]: List of retrieved triplets
        """  
        # Set up query parameters
        use_semantic = kwargs.get("use_semantic", True)
        use_graph = kwargs.get("use_graph", not use_semantic)
        similarity_threshold = kwargs.get("similarity_threshold", self.similarity_threshold)
        
        # Generate embedding for semantic search
        query_embedding = None
        if use_semantic:
            query_embedding = self._embedder.embed_query(query)
        
        # Run the async query
        results = asyncio.run(self._async_retrieve(
            query_text=query,
            query_embedding=query_embedding,
            use_semantic_search=use_semantic,
            use_graph_search=use_graph,
            limit=top_k,
            similarity_threshold=similarity_threshold
        ))
        
        # Format the results to match the BaseRAG expected format
        formatted_results = []
        for result in results:
            # Convert triplet to text format - chỉ hiển thị subject, predicate, object
            text = f"{result['subject']} {result['predicate']} {result['object']}"
            
            formatted_result = {
                "text": text,
                "content": text,
                "triplet": {
                    "subject": result.get("subject"),
                    "predicate": result.get("predicate"),
                    "object": result.get("object")
                },
                "metadata": {
                    "source": result.get("source", "knowledge_graph"),
                    "confidence": result.get("relevance_score", 1.0),
                    "seed_entity": result.get("seed_entity", ""),
                    "document_id": result.get("document_id", "")
                }
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents based on relevance to the query.
        
        Args:
            query (str): The original user query
            documents (List[Dict]): List of retrieved documents to rerank
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Dict[str, Any]]: Reranked documents with scores
        """
        # Simple reranking based on confidence/relevance score
        reranked = sorted(documents, key=lambda x: x.get("metadata", {}).get("confidence", 0), reverse=True)
        return reranked[:top_k]
    
    async def _async_retrieve(
        self,
        query_text: str,
        query_embedding: List[float] = None,
        use_semantic_search: bool = True,
        use_graph_search: bool = True,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous retrieval from knowledge graph
        
        Args:
            query_text: The query text
            query_embedding: Embedding vector for the query
            use_semantic_search: Whether to use semantic search
            use_graph_search: Whether to use graph-based search
            limit: Maximum number of results to return
            similarity_threshold: Threshold for semantic similarity
            
        Returns:
            List of matching triplets
        """
        semantic_results = []
        graph_results = []
        
        # Run searches in parallel
        search_tasks = []
        
        if use_semantic_search and query_embedding is not None:
            semantic_task = asyncio.create_task(
                self._perform_semantic_search(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    num_results=limit,
                    similarity_threshold=similarity_threshold
                )
            )
            search_tasks.append(semantic_task)
        
        if use_graph_search:
            graph_task = asyncio.create_task(
                self._perform_structured_search(
                    query_text,
                    num_results=limit
                )
            )
            search_tasks.append(graph_task)
        
        # Wait for all search tasks to complete
        search_results = await asyncio.gather(*search_tasks)
        
        # Assign results based on which searches were enabled
        if use_semantic_search and use_graph_search and query_embedding is not None:
            semantic_results = search_results[0]
            graph_results = search_results[1]
        elif use_semantic_search and query_embedding is not None:
            semantic_results = search_results[0]
        elif use_graph_search:
            graph_results = search_results[0]
        
        # Combine results if needed
        if self.hybrid_search and semantic_results and graph_results:
            combined_results = self._combine_search_results(
                semantic_results,
                graph_results,
                num_results=limit
            )
            return combined_results
        elif semantic_results:
            return semantic_results
        elif graph_results:
            return graph_results
        else:
            return []
    
    async def _find_similar_entities(
        self,
        query_embedding: List[float],
        limit: int = 10,
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
      
            
        # Perform vector search through Neo4j
        result = self._neo4j.run_vector_search(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        return result
    
    async def _semantic_entity_query(
        self,
        query_text: str,
        query_embedding: List[float],
        hops: int = None,
        limit: int = 3,
        similarity_threshold: float = 0.7
    ):
        """
        Perform semantic query based on vector embeddings and graph traversal
        
        Args:
            query_text: Original text query
            query_embedding: Embedding vector of the query
            hops: Number of graph traversal steps (default: self.max_hops)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of related triplets
        """
        
        hops = hops if hops is not None else self.max_hops
        
        # Find similar entities based on vector embedding
        similar_entities = await self._find_similar_entities(
            query_embedding,
            limit=5,  # Limit the number of seed entities
            similarity_threshold=similarity_threshold
        )

        if not similar_entities:
            return []
        
        # Extract entity names
        entity_names = [entity["entity"] for entity in similar_entities]
        
        # Query the graph to explore neighborhood
        try:
            # Simplified query that doesn't rely on APOC
            query = f"""
            MATCH (seed:Entity)
            WHERE seed.name IN $entity_names
            MATCH (s)-[r]->(o)
            WHERE (s)-[*0..{hops}]-(seed) OR (o)-[*0..{hops}]-(seed)
            AND type(r) <> 'FROM_DOCUMENT'
            RETURN DISTINCT
                s.name AS subject,
                r.name AS predicate,
                type(r) AS predicate_type,
                o.name AS object,
                seed.name AS seed_entity,
                CASE 
                    WHEN (s)-[*0..1]-(seed) AND (o)-[*0..1]-(seed) THEN 3
                    WHEN (s)-[*0..1]-(seed) OR (o)-[*0..1]-(seed) THEN 2
                    ELSE 1
                END AS relevance_score
            ORDER BY seed_entity, relevance_score DESC
            """
            
            params = {
                "entity_names": entity_names,
            }
            
            result = self._neo4j.execute_query(query, params)
            
            # Process results
            return [
                {
                    "subject": record["subject"],
                    "predicate": record.get("predicate") or record.get("predicate_type"),
                    "object": record["object"],
                    "seed_entity": record["seed_entity"],
                    "relevance_score": record["relevance_score"],
                    "source": "semantic"
                }
                for record in result[:limit]
            ]
        except Exception as e:
            logger.error(f"Error in semantic entity query with variable-length paths: {e}")
            return await self._fallback_entity_neighborhood_query(entity_names, limit)
    
    async def _fallback_entity_neighborhood_query(self, entity_names: List[str], limit: int = 20):
        """
        Fallback query for entity neighborhood when complex queries fail
        
        Args:
            entity_names: List of seed entity names
            limit: Maximum number of results
            
        Returns:
            List of simple neighborhood triplets
        """
        query = """
        MATCH (seed:Entity)-[r1]-(mid)
        WHERE seed.name IN $entity_names
        AND type(r1) <> 'FROM_DOCUMENT'
        RETURN DISTINCT
            seed.name AS subject,
            type(r1) AS predicate_type,
            r1.name AS predicate,
            mid.name AS object,
            seed.name AS seed_entity,
            1 AS relevance_score
        ORDER BY seed_entity
        LIMIT $limit
        """
        
        params = {
            "entity_names": entity_names,
            "limit": limit
        }
        
        result = self._neo4j.execute_query(query, params)
        
        # Không cần sắp xếp ở đây vì ORDER BY đã được áp dụng trong truy vấn
        
        return [
            {
                "subject": record["subject"],
                "predicate": record.get("predicate") or record.get("predicate_type"),
                "object": record["object"],
                "seed_entity": record["seed_entity"],
                "relevance_score": record["relevance_score"],
                "source": "semantic_fallback"
            }
            for record in result
        ]
        
    async def _perform_semantic_search(
        self,
        query_text: str,
        query_embedding: List[float],
        num_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using query embedding
        
        Args:
            query_text: The query text
            query_embedding: The query embedding vector
            num_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching triplets
        """
        # Use semantic entity query to search with vectors
        results = await self._semantic_entity_query(
            query_text=query_text,
            query_embedding=query_embedding,
            limit=num_results,
            similarity_threshold=similarity_threshold
        )
        # Sort results by entity first, then by relevance
        if results:
            results.sort(key=lambda x: (x.get("seed_entity", ""), -x.get("relevance_score", 0)))
            
        return results
    
    async def _perform_structured_search(
        self,
        query_text: str,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform structured search using text patterns
        
        Args:
            query_text: The query text
            num_results: Maximum number of results
            
        Returns:
            List of matching triplets
        """
        # Try direct keyword search (non-exact match)
        words = [w for w in query_text.split() if len(w) > 3]
        results = []
        
        for word in words:

            # Search for word as subject or object
            subject_results = self._neo4j.query_knowledge_graph(
                subject=word,
                limit=num_results // 2
            )
            
            object_results = self._neo4j.query_knowledge_graph(
                object_entity=word,
                limit=num_results // 2
            )
            
            # Add source information
            for result in subject_results + object_results:
                result["source"] = "structured"
                result["relevance_score"] = 0.8  # Default score for structured search
            
            results.extend(subject_results)
            results.extend(object_results)
            
            if len(results) >= num_results:
                break
        
        return results[:num_results]
    
    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        structured_results: List[Dict[str, Any]],
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Combine and rank semantic and structured search results
        
        Args:
            semantic_results: Results from semantic search
            structured_results: Results from structured search
            num_results: Maximum number of results to return
            
        Returns:
            Combined and ranked results
        """
        # Create a unique identifier for each triplet
        def create_triplet_id(triplet):
            subj = triplet.get("subject", "")
            pred = triplet.get("predicate", "")
            obj = triplet.get("object", "")
            return f"{subj}|{pred}|{obj}"
        
        # Create a map of triplet_id to result for deduplication
        combined_map = {}
        
        # Add semantic results with higher priority
        for result in semantic_results:
            # Kiểm tra xem predicate có phải là FROM_DOCUMENT không
            predicate = result.get("predicate", "")
            predicate_type = result.get("predicate_type", "")
            
            if predicate == "FROM_DOCUMENT" or predicate_type == "FROM_DOCUMENT":
                continue
                
            triplet_id = create_triplet_id(result)
            if triplet_id and triplet_id not in combined_map:
                combined_map[triplet_id] = result
        
        # Add structured results
        for result in structured_results:
            # Kiểm tra xem predicate có phải là FROM_DOCUMENT không
            predicate = result.get("predicate", "")
            predicate_type = result.get("predicate_type", "")
            
            if predicate == "FROM_DOCUMENT" or predicate_type == "FROM_DOCUMENT":
                continue
                
            triplet_id = create_triplet_id(result)
            if triplet_id and triplet_id not in combined_map:
                combined_map[triplet_id] = result
        
        # Convert back to list and sort by relevance score
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Limit results
        return combined_results[:num_results] 