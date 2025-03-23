import logging
import concurrent.futures
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

from retrieval_engine.knowledge_retrieval.base_rag import BaseRAG
from retrieval_engine.knowledge_retrieval.vector_rag.vector_rag import VectorRAG
from retrieval_engine.knowledge_retrieval.graph_rag_v2.graph_rag import GraphRAG

logger = logging.getLogger(__name__)

class HybridRAG(BaseRAG):
    """
    Hybrid Retrieval Augmented Generation system that combines outputs 
    from both Vector-based and Graph-based RAG systems.
    
    This class implements different strategies for combining results from
    multiple retrieval methods to provide more comprehensive and accurate results.
    """
    
    def __init__(
        self,
        vector_rag: Optional[VectorRAG] = None,
        graph_rag: Optional[GraphRAG] = None,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
        combination_strategy: str = "weighted",
        model_name: Optional[str] = None,
        deduplicate: bool = True,
        max_workers: int = 2
    ):
        """
        Initialize the Hybrid RAG system.
        
        Args:
            vector_rag (Optional[VectorRAG]): Vector RAG instance
            graph_rag (Optional[GraphRAG]): Graph RAG v2 instance
            vector_weight (float): Weight for vector results (0.0 to 1.0)
            graph_weight (float): Weight for graph results (0.0 to 1.0)
            combination_strategy (str): How to combine results ('weighted', 'ensemble', 'cascade')
            model_name (Optional[str]): Model name for reference
            deduplicate (bool): Whether to remove duplicate results
            max_workers (int): Maximum number of parallel workers
        """
        super().__init__(model_name=model_name)
        
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag
        
        # Validate weights
        assert 0 <= vector_weight <= 1, "Vector weight must be between 0 and 1"
        assert 0 <= graph_weight <= 1, "Graph weight must be between 0 and 1"
        
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.combination_strategy = combination_strategy
        self.deduplicate = deduplicate
        self.max_workers = max_workers
        
        # Validate we have at least one RAG system
        if not vector_rag and not graph_rag:
            raise ValueError("At least one RAG system (vector or graph) must be provided")
        
        logger.info(f"Initialized HybridRAG with strategy '{combination_strategy}' and {max_workers} max workers")
    
    def _retrieve_from_vector(self, query: str, top_k: int, vector_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Helper method to retrieve from Vector RAG with error handling."""
        try:
            logger.info("Retrieving from Vector RAG...")
            results = self.vector_rag.retrieve(query, top_k=top_k, **vector_kwargs)
            logger.info(f"Vector RAG returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error retrieving from Vector RAG: {str(e)}")
            return []
    
    def _retrieve_from_graph(self, query: str, graph_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Helper method to retrieve from Graph RAG with error handling."""
        try:
            logger.info("Retrieving from Graph RAG...")
            results = self.graph_rag.retrieve(query, **graph_kwargs)
            normalized_results = self._normalize_graph_results(results)
            logger.info(f"Graph RAG returned {len(normalized_results)} results")
            return normalized_results
        except Exception as e:
            logger.error(f"Error retrieving from Graph RAG: {str(e)}")
            return []
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using both Vector and Graph RAG systems in parallel.
        
        Args:
            query (str): The user query
            top_k (int): Number of top results to retrieve
            **kwargs: Additional parameters for retrieval
                - vector_kwargs: Dictionary of arguments specific to Vector RAG
                - graph_kwargs: Dictionary of arguments specific to Graph RAG
            
        Returns:
            List[Dict[str, Any]]: Combined results from both systems
        """
        vector_results = []
        graph_results = []
        
        # Extract kwargs for each RAG system
        vector_kwargs = kwargs.get("vector_kwargs", {})
        graph_kwargs = kwargs.get("graph_kwargs", {})
        
        # Determine which RAG systems to use
        use_vector = self.vector_rag is not None
        use_graph = self.graph_rag is not None
        
        # Use parallel execution with ThreadPoolExecutor
        if use_vector and use_graph:
            logger.info("Running Vector and Graph retrievals in parallel")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit both retrieval tasks
                vector_future = executor.submit(
                    self._retrieve_from_vector, query, top_k, vector_kwargs
                )
                graph_future = executor.submit(
                    self._retrieve_from_graph, query, graph_kwargs
                )
                
                # Get results as they complete
                for future in concurrent.futures.as_completed([vector_future, graph_future]):
                    try:
                        result = future.result()
                        if future == vector_future:
                            vector_results = result
                        elif future == graph_future:
                            graph_results = result
                    except Exception as e:
                        logger.error(f"Error in parallel retrieval: {str(e)}")
        else:
            # Fall back to sequential if only one system is available
            if use_vector:
                vector_results = self._retrieve_from_vector(query, top_k, vector_kwargs)
            if use_graph:
                graph_results = self._retrieve_from_graph(query, graph_kwargs)
        
        # Combine results based on strategy
        combined_results = self._combine_results(
            query=query,
            vector_results=vector_results, 
            graph_results=graph_results, 
            top_k=top_k
        )
        
        return combined_results
    
    def _normalize_graph_results(self, graph_results: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize graph results to match the expected format.
        
        Args:
            graph_results: Results from graph RAG
            
        Returns:
            List[Dict[str, Any]]: Normalized results
        """
        normalized = []
        
        for i, result in enumerate(graph_results):
            # GraphRAG v2 đã có format gần với format chuẩn, chỉ cần thêm trường score
            if "text" in result and "content" in result and "metadata" in result:
                # Format mới của GraphRAG v2
                normalized_result = result.copy()
                # Đưa relevance_score từ metadata thành score chính cho việc sắp xếp
                normalized_result["score"] = result["metadata"].get("confidence", 1.0)
                normalized.append(normalized_result)
            else:
                # Fallback cho format khác
                normalized.append({
                    "content": result if isinstance(result, str) else str(result),
                    "text": result if isinstance(result, str) else str(result),
                    "metadata": {
                        "source": "graph",
                        "result_id": f"graph_{i}",
                        "confidence": 1.0
                    },
                    "score": 1.0  # Default score
                })
        
        return normalized
    
    def _combine_results(
        self, 
        query: str,
        vector_results: List[Dict[str, Any]], 
        graph_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results from vector and graph RAG systems.
        
        Args:
            query: The original query
            vector_results: Results from vector RAG
            graph_results: Results from graph RAG
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        if self.combination_strategy == "weighted":
            return self._weighted_combination(vector_results, graph_results, top_k)
        elif self.combination_strategy == "ensemble":
            return self._ensemble_combination(vector_results, graph_results, top_k)
        elif self.combination_strategy == "cascade":
            return self._cascade_combination(query, vector_results, graph_results, top_k)
        else:
            logger.warning(f"Unknown combination strategy: {self.combination_strategy}. Using weighted.")
            return self._weighted_combination(vector_results, graph_results, top_k)
    
    def _weighted_combination(
        self, 
        vector_results: List[Dict[str, Any]], 
        graph_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using a weighted approach.
        
        Args:
            vector_results: Results from vector RAG
            graph_results: Results from graph RAG
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        # Normalize scores within each result set
        for results in [vector_results, graph_results]:
            self._normalize_scores(results)
        
        # Apply weights to each result set
        for result in vector_results:
            result["score"] = result.get("score", 0.0) * self.vector_weight
            result["metadata"]["source_type"] = "vector"
        
        for result in graph_results:
            result["score"] = result.get("score", 0.0) * self.graph_weight
            result["metadata"]["source_type"] = "graph"
        
        # Combine results
        all_results = vector_results + graph_results
        
        # Deduplicate if needed
        if self.deduplicate:
            all_results = self._deduplicate_results(all_results)
        
        # Sort by score and take top_k
        sorted_results = sorted(all_results, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_results[:top_k]
    
    def _ensemble_combination(
        self, 
        vector_results: List[Dict[str, Any]], 
        graph_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using an ensemble approach (interleaving).
        
        Args:
            vector_results: Results from vector RAG
            graph_results: Results from graph RAG
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        # Add source metadata
        for result in vector_results:
            result["metadata"]["source_type"] = "vector"
        
        for result in graph_results:
            result["metadata"]["source_type"] = "graph"
        
        # Interleave results
        combined = []
        max_length = max(len(vector_results), len(graph_results))
        
        for i in range(max_length):
            if i < len(vector_results):
                combined.append(vector_results[i])
            if i < len(graph_results):
                combined.append(graph_results[i])
        
        # Deduplicate if needed
        if self.deduplicate:
            combined = self._deduplicate_results(combined)
            
        return combined[:top_k]
    
    def _cascade_combination(
        self,
        query: str,
        vector_results: List[Dict[str, Any]], 
        graph_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using a cascade approach (try vector first, then graph).
        
        Args:
            query: The original query
            vector_results: Results from vector RAG
            graph_results: Results from graph RAG
            top_k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        # Add source metadata
        for result in vector_results:
            result["metadata"]["source_type"] = "vector"
        
        for result in graph_results:
            result["metadata"]["source_type"] = "graph"
        
        # If vector results are sufficient, return them
        if len(vector_results) >= top_k:
            # Check quality of vector results (simple heuristic based on score)
            scores = [r.get("score", 0.0) for r in vector_results[:top_k]]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            if avg_score > 0.7:  # Threshold for "good enough" vector results
                return vector_results[:top_k]
        
        # Otherwise, combine vector and graph results
        combined = vector_results + graph_results
        
        # Deduplicate if needed
        if self.deduplicate:
            combined = self._deduplicate_results(combined)
            
        return combined[:top_k]
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> None:
        """
        Normalize scores within a result set to range [0, 1].
        
        Args:
            results: List of result documents
        """
        if not results:
            return
            
        # Find min and max scores
        scores = [r.get("score", 0.0) for r in results]
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        
        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result["score"] = 1.0 if "score" in result else 0.0
            return
        
        # Normalize scores
        for result in results:
            if "score" in result:
                result["score"] = (result["score"] - min_score) / (max_score - min_score)
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on content similarity.
        
        Args:
            results: List of result documents
            
        Returns:
            List[Dict[str, Any]]: Deduplicated results
        """
        deduplicated = []
        content_set = set()
        
        for result in results:
            content = result.get("content", "").strip()
            # Use a substring for deduplication to handle minor differences
            content_key = content[:100].lower()
            
            if content_key not in content_set:
                content_set.add(content_key)
                deduplicated.append(result)
        
        return deduplicated
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using the best available reranker.
        
        Args:
            query (str): The user query
            documents (List[Dict]): Documents to rerank
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Reranked documents
        """
        # Prefer vector reranker if available
        if self.vector_rag:
            return self.vector_rag.rerank(query, documents, top_k)
        
        # Otherwise, just sort by score and return top_k
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_docs[:top_k]
    
    async def close(self):
        """Close all connections from RAG systems."""
        if hasattr(self.graph_rag, 'close') and callable(self.graph_rag.close):
            try:
                await self.graph_rag.close()
                logger.info("Closed GraphRAG connections")
            except Exception as e:
                logger.error(f"Error closing GraphRAG connections: {e}")
        
        if hasattr(self.vector_rag, 'close') and callable(self.vector_rag.close):
            try:
                if asyncio.iscoroutinefunction(self.vector_rag.close):
                    await self.vector_rag.close()
                else:
                    self.vector_rag.close()
                logger.info("Closed VectorRAG connections")
            except Exception as e:
                logger.error(f"Error closing VectorRAG connections: {e}")
        
        logger.info("HybridRAG connections closed") 