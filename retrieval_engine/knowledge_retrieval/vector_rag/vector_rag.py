import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from retrieval_engine.knowledge_retrieval.base_rag import BaseRAG
from retrieval_engine.knowledge_retrieval.vector_rag.qdrant_connection import VectorStore
from retrieval_engine.knowledge_retrieval.vector_rag.reranker import DocumentReranker, RerankMethod
from retrieval_engine.knowledge_retrieval.vector_rag.embeddings import (
    EmbeddingProvider, 
    EmbeddingFactory
)
from retrieval_engine.knowledge_retrieval.abs_cls import SingletonMeta

logger = logging.getLogger(__name__)

class VectorRAG(BaseRAG):
    """
    Vector-based Retrieval Augmented Generation using Qdrant vector database.
    
    This class implements the RAG pattern using vector similarity search
    for knowledge retrieval and supports multiple reranking strategies.
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        embedding_provider: Optional[str] = "huggingface",
        embedding_model_name: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        reranker_strategy: Union[str, RerankMethod] = RerankMethod.CROSS_ENCODER,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        qdrant_grpc_port: Optional[int] = None,
        qdrant_api_key: Optional[str] = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the Vector RAG system.
        
        Args:
            collection_name (str): Name of the Qdrant collection to use
            embedding_provider (Optional[str]): Provider for embeddings (huggingface, openai, google)
            embedding_model_name (Optional[str]): Name of the embedding model
            embedding_api_key (Optional[str]): API key for the embedding provider
            reranker_model_name (Optional[str]): Name of the reranker model
            reranker_strategy (Union[str, RerankMethod]): Strategy for reranking
            qdrant_host (Optional[str]): Qdrant server host
            qdrant_port (Optional[int]): Qdrant server port
            qdrant_grpc_port (Optional[int]): Qdrant server gRPC port
            qdrant_api_key (Optional[str]): Qdrant API key
            cache_embeddings (bool): Whether to cache embeddings
        """
        super().__init__(model_name=embedding_model_name)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            host=qdrant_host,
            port=qdrant_port, 
        )
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_provider = embedding_provider
        self.cache_embeddings = cache_embeddings
        self._init_embedding_model(embedding_api_key)
        
        # Initialize reranker
        self.reranker = DocumentReranker(
            strategy=reranker_strategy,
            model_name=reranker_model_name
        )
        
        logger.info(f"Initialized VectorRAG with collection '{collection_name}' and {embedding_provider} embeddings")
    
    def _init_embedding_model(self, api_key: Optional[str] = None):
        """
        Initialize the embedding model for query encoding.
        
        Args:
            api_key (Optional[str]): API key for embedding provider 
        """
        try:
            model_name = self.model_name
            provider_type = self.embedding_provider.lower()
            
            # Set default model names based on provider if not specified
            if not model_name:
                if provider_type == "openai":
                    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                elif provider_type == "google":
                    model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
                else:
                    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            
            # Create embedding provider using factory
            self.embedding_model = EmbeddingFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key,
                cache=self.cache_embeddings
            )
            
            logger.info(f"Loaded {provider_type} embedding model: {model_name}")
            
            # Get vector dimension for potential collection creation
            self.vector_dim = self.embedding_model.dimension
            logger.info(f"Embedding dimension: {self.vector_dim}")
            
        except ImportError as e:
            logger.error(f"Dependencies not installed: {str(e)}")
            raise ImportError(f"Required packages not installed: {str(e)}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into a vector representation.
        
        Args:
            query (str): The query text to encode
            
        Returns:
            np.ndarray: The query vector embedding
        """
        return self.embedding_model.embed_query(query)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode a batch of documents into vector representations.
        
        Args:
            documents (List[str]): List of document texts to encode
            
        Returns:
            np.ndarray: The document vector embeddings
        """
        return self.embedding_model.embed_documents(documents)
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> None:
        """
        Index documents into the vector store.
        
        Args:
            documents (List[str]): List of document texts to index
            metadatas (Optional[List[Dict]]): List of metadata for each document
            ids (Optional[List[str]]): List of IDs for the documents
            batch_size (int): Batch size for encoding and indexing
            
        Returns:
            None
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
            
        # Create collection if it doesn't exist
        try:
            # Check if collection exists by trying to retrieve info
            self.vector_store.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection '{self.collection_name}' not found: {str(e)}")
            logger.info(f"Creating new collection: {self.collection_name}")
            self.vector_store.create_collection(
                collection_name=self.collection_name,
                vector_size=self.vector_dim
            )
        
        # Prepare metadata if not provided
        if metadatas is None:
            metadatas = [{"text": doc} for doc in documents]
        
        # Process documents in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            batch_ids = None if ids is None else ids[i:i+batch_size]
            
            # Encode the batch
            batch_vectors = self.encode_documents(batch_docs)
            
            # Ensure text is in metadata
            for j, meta in enumerate(batch_metadata):
                if "text" not in meta:
                    meta["text"] = batch_docs[j]
            
            # Add embeddings to vector store
            self.vector_store.insert_vectors(
                collection_name=self.collection_name,
                vectors=batch_vectors,
                payloads=batch_metadata,
                ids=batch_ids
            )
            
            logger.info(f"Indexed batch of {len(batch_docs)} documents")
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query (str): The query to search for
            top_k (int): Number of results to retrieve
            **kwargs: Additional parameters for retrieval
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents
        """
        # Extract filter conditions if provided
        filter_conditions = kwargs.get("filter_conditions")
        score_threshold = kwargs.get("score_threshold")
        
        try:
            # Encode the query
            query_vector = self.encode_query(query)
            
            # Perform retrieval
            if filter_conditions:
                # Use filtered search
                results = self.vector_store.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    filter_conditions=filter_conditions,
                    limit=top_k
                )
            else:
                # Use standard search
                results = self.vector_store.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold
                )
            
            # Format results for consistency
            documents = []
            for result in results:
                # Extract content from payload
                content = result["payload"].get("text", "")
                
                # Create formatted document
                document = {
                    "id": result["id"],
                    "content": content,
                    "score": result["score"],
                    "metadata": {k: v for k, v in result["payload"].items() if k != "text"}
                }
                documents.append(document)
                
            logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using the configured reranker.
        
        Args:
            query (str): The original query
            documents (List[Dict]): The documents to rerank
            top_k (int): Number of top results to keep
            
        Returns:
            List[Dict[str, Any]]: Reranked documents
        """
        try:
            reranked_docs = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            logger.info(f"Reranked {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            # If reranking fails, return the original documents
            return documents[:top_k]
    