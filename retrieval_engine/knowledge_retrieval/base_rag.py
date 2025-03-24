import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class BaseRAG(ABC):
    """
    Abstract base class for RAG (Retrieval Augmented Generation) pipelines.
    
    This class defines the common interface and shared functionality for different
    types of RAG implementations (Vector-based, Graph-based, etc.).
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the base RAG pipeline.
        
        Args:
            model_name (Optional[str]): Name of the embedding model to use (if applicable)
        """
        self.model_name = model_name
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query (str): The user query to search for relevant documents
            top_k (int): Number of top results to retrieve
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with metadata
        """
        pass
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents based on relevance to the query.
        
        Args:
            query (str): The original user query
            documents (List[Dict]): List of retrieved documents to rerank
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Dict[str, Any]]: Reranked documents with scores
        """
        pass
    
    def process(self, query: str, top_k: int = 5, rerank: bool = True, rerank_top_k: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline: retrieval + optional reranking.
        
        Args:
            query (str): The user query
            top_k (int): Number of results to retrieve
            rerank (bool): Whether to rerank the retrieved results
            rerank_top_k (Optional[int]): Number of results to keep after reranking
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Dict[str, Any]: Results containing retrieved documents and metadata
        """
        logger.info(f"Processing query: '{query}'")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k, **kwargs)
        
        if not retrieved_docs:
            logger.warning(f"No documents retrieved for query: '{query}'")
            return {
                "query": query,
                "documents": [],
                "reranked": False,
                "metadata": {"total_results": 0}
            }
        
        # Apply reranking if enabled
        if rerank:
            rerank_limit = rerank_top_k or top_k
            reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_limit)
            
            return {
                "query": query,
                "documents": reranked_docs,
                "reranked": True,
                "metadata": {
                    "total_results": len(retrieved_docs),
                    "returned_results": len(reranked_docs)
                }
            }
        
        return {
            "query": query,
            "documents": retrieved_docs,
            "reranked": False,
            "metadata": {
                "total_results": len(retrieved_docs),
                "returned_results": len(retrieved_docs)
            }
        }
    
    def format_context(self, results: Dict[str, Any], include_metadata: bool = False) -> str:
        """
        Format retrieval results into a context string for the LLM.
        
        Args:
            results (Dict[str, Any]): Results from the process method
            include_metadata (bool): Whether to include document metadata in the context
            
        Returns:
            str: Formatted context string for the LLM
        """
        if not results.get("documents"):
            return "No relevant information found."
        
        context_parts = []
        
        for i, doc in enumerate(results["documents"], 1):
            # Extract content from the document
            content = doc.get("content", doc.get("text", ""))
            
            # Format the document with its index
            doc_section = f"[Document {i}]: {content}"
            
            # Add metadata if requested
            if include_metadata and "metadata" in doc:
                meta_str = ", ".join(f"{k}: {v}" for k, v in doc["metadata"].items()
                                    if k not in ["content", "text"])
                doc_section += f"\n[Metadata]: {meta_str}"
            
            context_parts.append(doc_section)
        
        # Join all document sections
        return "\n\n".join(context_parts) 