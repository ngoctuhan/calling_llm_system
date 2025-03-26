import os
import logging
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from enum import Enum, auto
from retrieval_engine.knowledge_retrieval.abs_cls import SingletonMeta
logger = logging.getLogger(__name__)

class RerankMethod(str, Enum):
    """Enum representing different reranking methods"""
    CROSS_ENCODER = "cross_encoder"
    BM25 = "bm25"
    MMR = "mmr"
    DEFAULT = "default"
    
    @classmethod
    def from_string(cls, value: str) -> "RerankMethod":
        """Convert string to enum value, case-insensitive"""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.DEFAULT

class DocumentReranker(metaclass=SingletonMeta):
    """
    Document reranker for improving RAG retrieval results.
    
    This class implements various strategies for reranking retrieved documents
    to improve their relevance to the user query.
    """
    
    def __init__(self, strategy: Union[str, RerankMethod] = RerankMethod.CROSS_ENCODER, 
                 model_name: Optional[str] = None):
        """
        Initialize the document reranker.
        
        Args:
            strategy (Union[str, RerankMethod]): Reranking strategy to use. Options:
                - "cross_encoder": Uses a cross-encoder model for precise relevance scoring
                - "bm25": Uses BM25 text similarity algorithm
                - "mmr": Uses Maximum Marginal Relevance for relevance and diversity
                - "default": Uses the original scores
            model_name (Optional[str]): Name of the model to use (for strategies that require it)
        """
        if isinstance(strategy, str):
            self.strategy = RerankMethod.from_string(strategy)
        else:
            self.strategy = strategy
            
        self.model_name = model_name
        self._init_reranker()
        
        logger.info(f"Initialized DocumentReranker with strategy: {self.strategy.value}")
    
    def _init_reranker(self):
        """Initialize the underlying reranker model based on the selected strategy."""
        if self.strategy == RerankMethod.CROSS_ENCODER:
            try:
                from sentence_transformers import CrossEncoder
                
                model = self.model_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.model = CrossEncoder(model)
                logger.info(f"Loaded CrossEncoder model: {model}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Using default similarity scoring.")
                self.strategy = RerankMethod.DEFAULT
                
        elif self.strategy == RerankMethod.BM25:
            try:
                import rank_bm25
                self.bm25_tokenizer = lambda x: x.lower().split()
                logger.info("Initialized BM25 reranker")
            except ImportError:
                logger.warning("rank_bm25 not installed. Using default similarity scoring.")
                self.strategy = RerankMethod.DEFAULT
                
        elif self.strategy == RerankMethod.MMR:
            # For MMR we need an embedding function
            try:
                ...
            except ImportError:
                logger.warning("sentence-transformers not installed. Using default similarity scoring.")
                self.strategy = RerankMethod.DEFAULT
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5, 
               diversity_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Rerank the retrieved documents based on the chosen strategy.
        
        Args:
            query (str): The user query
            documents (List[Dict]): The retrieved documents to rerank
            top_k (int): Number of top results to return
            diversity_weight (float): Weight for diversity in MMR (0.0-1.0, higher means more diverse)
            
        Returns:
            List[Dict[str, Any]]: Reranked documents with updated scores
        """
        if not documents:
            return []
            
        # Limit to the actual number of documents if top_k is larger
        top_k = min(top_k, len(documents))
        
        if self.strategy == RerankMethod.CROSS_ENCODER:
            return self._rerank_cross_encoder(query, documents, top_k)
        elif self.strategy == RerankMethod.BM25:
            return self._rerank_bm25(query, documents, top_k)
        elif self.strategy == RerankMethod.MMR:
            return self._rerank_mmr(query, documents, top_k, diversity_weight)
        else:
            # Default strategy uses the original scores
            return sorted(documents, key=lambda x: -x.get("score", 0))[:top_k]
    
    def _rerank_cross_encoder(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using a cross-encoder model."""
        # Extract document texts
        doc_texts = [doc.get("content", doc.get("payload", {}).get("text", "")) for doc in documents]
        
        # Create query-document pairs for scoring
        query_doc_pairs = [[query, text] for text in doc_texts]
        
        # Score all the pairs
        scores = self.model.predict(query_doc_pairs)
        
        # Annotate documents with new scores
        reranked_docs = []
        for i, (score, doc) in enumerate(zip(scores, documents)):
            doc_copy = doc.copy()
            doc_copy["original_score"] = doc_copy.get("score", 0.0)
            doc_copy["score"] = float(score)
            doc_copy["rank"] = i + 1
            reranked_docs.append(doc_copy)
        
        # Sort by the new scores and return top_k
        reranked_docs.sort(key=lambda x: -x["score"])
        return reranked_docs[:top_k]
    
    def _rerank_bm25(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using BM25 algorithm."""
        from rank_bm25 import BM25Okapi
        
        # Extract document texts
        doc_texts = [doc.get("content", doc.get("payload", {}).get("text", "")) for doc in documents]
        
        # Tokenize documents
        tokenized_docs = [self.bm25_tokenizer(text) for text in doc_texts]
        tokenized_query = self.bm25_tokenizer(query)
        
        # Create BM25 model and get scores
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        
        # Annotate documents with new scores
        reranked_docs = []
        for i, (score, doc) in enumerate(zip(scores, documents)):
            doc_copy = doc.copy()
            doc_copy["original_score"] = doc_copy.get("score", 0.0)
            doc_copy["score"] = float(score)
            doc_copy["rank"] = i + 1
            reranked_docs.append(doc_copy)
        
        # Sort by the new scores and return top_k
        reranked_docs.sort(key=lambda x: -x["score"])
        return reranked_docs[:top_k]
    
    def _rerank_mmr(self, query: str, documents: List[Dict[str, Any]], top_k: int,
                   diversity_weight: float) -> List[Dict[str, Any]]:
        """Rerank using Maximum Marginal Relevance for diversity and relevance."""
        # Extract document texts
        raise NotImplementedError("MMR reranking is not implemented yet")
    
    def evaluate(self, queries: List[str], all_documents: List[List[Dict[str, Any]]],
                relevant_ids: List[List[str]], metrics: List[str] = ["precision", "recall", "ndcg"]) -> Dict[str, float]:
        """
        Evaluate the reranker using standard information retrieval metrics.
        
        Args:
            queries (List[str]): List of evaluation queries
            all_documents (List[List[Dict]]): List of document sets for each query
            relevant_ids (List[List[str]]): List of relevant document IDs for each query
            metrics (List[str]): Metrics to compute
            
        Returns:
            Dict[str, float]: Evaluation results
        """
        results = {metric: 0.0 for metric in metrics}
        
        for query, documents, rel_ids in zip(queries, all_documents, relevant_ids):
            # Rerank the documents
            reranked = self.rerank(query, documents)
            
            # Get the reranked IDs
            reranked_ids = [doc.get("id", doc.get("payload", {}).get("id", "")) for doc in reranked]
            
            # Compute the requested metrics
            for metric in metrics:
                if metric == "precision":
                    # Precision@k = relevant_retrieved / retrieved
                    precision = len(set(reranked_ids) & set(rel_ids)) / len(reranked_ids) if reranked_ids else 0
                    results["precision"] += precision
                    
                elif metric == "recall":
                    # Recall = relevant_retrieved / relevant
                    recall = len(set(reranked_ids) & set(rel_ids)) / len(rel_ids) if rel_ids else 0
                    results["recall"] += recall
                    
                elif metric == "ndcg":
                    # NDCG (Normalized Discounted Cumulative Gain)
                    idcg = sum((1 / np.log2(i + 2)) for i in range(min(len(rel_ids), len(reranked_ids))))
                    dcg = sum((1 / np.log2(i + 2)) for i, doc_id in enumerate(reranked_ids) if doc_id in rel_ids)
                    ndcg = dcg / idcg if idcg > 0 else 0
                    results["ndcg"] += ndcg
        
        # Compute averages
        num_queries = len(queries)
        for metric in results:
            results[metric] /= num_queries
            
        return results 

# Example usage
if __name__ == "__main__":
    # Sample documents in Qdrant format returned from vector store
    example_documents = [
        {
            "id": "2a443064-2fba-4ea1-a2df-d3a5ad161383",
            "version": 0,
            "score": 0.89,
            "payload": {
                "id": "doc1",
                "text": "Call centers use AI-powered chatbots to handle simple customer queries.",
                "metadata": {"source": "customer_service_guide.pdf", "page": 12}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        },
        {
            "id": "3b554175-3cba-5eb2-b3df-e4b6bd272394",
            "version": 0,
            "score": 0.85,
            "payload": {
                "id": "doc2",
                "text": "AI systems can analyze customer sentiment during calls.",
                "metadata": {"source": "ai_applications.pdf", "page": 45}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        },
        {
            "id": "4c665286-4dcb-6fc3-c4ef-f5c7ce383405",
            "version": 0,
            "score": 0.82,
            "payload": {
                "id": "doc3",
                "text": "Call center agents use knowledge bases to answer customer questions quickly.",
                "metadata": {"source": "training_manual.pdf", "page": 23}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        },
        {
            "id": "5d776397-5edc-7fd4-d5fg-g6d8df494516",
            "version": 0,
            "score": 0.78,
            "payload": {
                "id": "doc4",
                "text": "Speech recognition technologies transcribe call center conversations automatically.",
                "metadata": {"source": "tech_implementations.pdf", "page": 67}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        },
        {
            "id": "6e887408-6fed-8ge5-e6gh-h7e9eg505627",
            "version": 0,
            "score": 0.75,
            "payload": {
                "id": "doc5",
                "text": "Call routing systems direct customers to the most appropriate agent or department.",
                "metadata": {"source": "call_center_operations.pdf", "page": 34}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        },
        {
            "id": "7f998519-7gfe-9hf6-f7hi-i8f0fh616738",
            "version": 0,
            "score": 0.70,
            "payload": {
                "id": "doc6",
                "text": "Natural language processing helps understand customer inquiries.",
                "metadata": {"source": "nlp_applications.pdf", "page": 56}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        },
        {
            "id": "8g009620-8hgf-0ig7-g8ij-j9g1gi727849",
            "version": 0,
            "score": 0.65,
            "payload": {
                "id": "doc7",
                "text": "Machine learning algorithms predict common customer issues based on historical data.",
                "metadata": {"source": "predictive_analytics.pdf", "page": 29}
            },
            "vector": None,
            "shard_key": "",
            "order_value": ""
        }
    ]
    
    # Sample query
    query = "How do call centers use AI for customer service?"
    
    print(f"Query: {query}")
    print(f"Original ranking of documents:")
    for i, doc in enumerate(example_documents):
        print(f"{i+1}. [{doc['score']:.3f}] {doc['payload']['text']}")
    
    print("\n" + "="*80 + "\n")
    
    # Try each reranking method
    reranking_methods = [
        RerankMethod.CROSS_ENCODER,
        RerankMethod.BM25,
        # RerankMethod.MMR,
        RerankMethod.DEFAULT
    ]

    # Fix document content access before reranking
    def _fix_doc_content_access(document_list):
        """
        Modify document structure to handle Qdrant format for reranking.
        Make a deep copy to avoid modifying the originals.
        """
        import copy
        fixed_docs = copy.deepcopy(document_list)
        for doc in fixed_docs:
            # Add content field expected by reranker methods
            doc["content"] = doc["payload"]["text"]
            # Add id field expected by evaluation
            doc["id"] = doc["payload"]["id"]
        return fixed_docs
    
    for method in reranking_methods:
        try:
            print(f"\nReranking using {method.value}:")
            
            # Initialize the reranker with the current method
            reranker = DocumentReranker(strategy=method)
            
            # Fix document structure for reranking
            fixed_docs = _fix_doc_content_access(example_documents)
            
            # Rerank the documents
            reranked_docs = reranker.rerank(
                query=query,
                documents=fixed_docs,
                top_k=5,
                diversity_weight=0.3 if method == RerankMethod.MMR else 0.0
            )
            
            # Display the reranked results
            for i, doc in enumerate(reranked_docs):
                original_score = doc.get("original_score", "N/A")
                new_score = doc.get("score", "N/A")
                print(f"{i+1}. [New: {new_score:.3f}, Orig: {original_score:.3f}] {doc['payload']['text']}")
                
        except Exception as e:
            print(f"Error with {method.value} reranker: {str(e)}")
            
    print("\n" + "="*80)
    
    # Example of evaluation
    print("\nEvaluation Example:")
    
    # Define evaluation data
    eval_queries = [
        "How do call centers use AI?",
        "What technologies are used in call centers?"
    ]
    
    # Fix document structure for evaluation
    fixed_example_docs = _fix_doc_content_access(example_documents)
    
    eval_documents = [
        fixed_example_docs,  # For first query
        fixed_example_docs   # For second query (using same docs for simplicity)
    ]
    
    # Define relevant document IDs for each query (ground truth)
    relevant_ids = [
        ["doc1", "doc2", "doc6"],  # Relevant docs for first query
        ["doc3", "doc4", "doc5"]   # Relevant docs for second query
    ]
    
    # Choose a reranker for evaluation
    eval_reranker = DocumentReranker(strategy=RerankMethod.CROSS_ENCODER)
    
    # Run evaluation
    eval_results = eval_reranker.evaluate(
        queries=eval_queries,
        all_documents=eval_documents,
        relevant_ids=relevant_ids,
        metrics=["precision", "recall", "ndcg"]
    )
    
    # Display evaluation results
    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"  {metric.capitalize()}: {value:.4f}") 