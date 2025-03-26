import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance, 
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    QueryRequest
)

logger = logging.getLogger(__name__)

class VectorStore:
    """
    A client for interacting with Qdrant vector database to store and retrieve vectors.
    
    This class provides a high-level interface for vector search operations, including
    creating collections, inserting vectors, and performing semantic searches.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        Initialize the Qdrant vector store connection.
        
        Args:
            host (Optional[str]): Qdrant server host. If None, uses QDRANT_HOST env variable.
            port (Optional[int]): Qdrant server port. If None, uses QDRANT_PORT env variable.
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        
        logger.info(f"Initializing Qdrant connection to {self.host}:{self.port}")
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=5.0
        )
        
    def get_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection info if it exists.
        
        Args:
            collection_name (str): The name of the collection to check.
            
        Returns:
            Dict[str, Any]: Collection information.
            
        Raises:
            Exception: If the collection does not exist.
        """
        logger.info(f"Checking if collection '{collection_name}' exists")
        return self.client.get_collection(collection_name=collection_name)
        
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name (str): The name of the collection to check.
            
        Returns:
            bool: True if the collection exists, False otherwise.
        """
        try:
            self.get_collection(collection_name)
            return True
        except Exception:
            return False
        
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "COSINE") -> None:
        """
        Create a new collection in Qdrant with the specified name and vector size.

        Args:
            collection_name (str): The name of the collection to be created.
            vector_size (int): The size of the vectors to be stored in the collection.
            distance (str): Distance metric to use ("COSINE", "EUCLID", or "DOT").

        Returns:
            None
        """
        distance_map = {
            "COSINE": Distance.COSINE,
            "EUCLID": Distance.EUCLID,
            "DOT": Distance.DOT
        }
        
        distance_metric = distance_map.get(distance, Distance.COSINE)
        
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_metric),
        )
        logger.info(f"Collection '{collection_name}' created successfully")
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name (str): The name of the collection to delete.
            
        Returns:
            None
        """
        logger.info(f"Deleting collection '{collection_name}'")
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully")

    def insert_vectors(self, 
                      collection_name: str, 
                      vectors: np.ndarray, 
                      payloads: List[Dict[str, Any]], 
                      ids: Optional[List[str]] = None) -> None:
        """
        Insert vectors into the specified collection in Qdrant.
        
        Args:
            collection_name (str): The name of the collection to insert vectors into.
            vectors (np.ndarray): A numpy array of vectors to be inserted.
            payloads (List[Dict]): A list of payloads corresponding to each vector.
            ids (Optional[List[str]]): Optional list of IDs for the vectors. If None, UUIDs will be generated.
            
        Returns:
            None
        """
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors and payloads must match")
            
        if ids is not None and len(ids) != len(vectors):
            raise ValueError("If provided, number of IDs must match number of vectors")
            
        points = []
        for idx, vector in enumerate(vectors):
            point_id = ids[idx] if ids is not None else uuid.uuid4().hex
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payloads[idx]
                )
            )
            
        logger.info(f"Inserting {len(points)} vectors into collection '{collection_name}'")
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully inserted {len(points)} vectors")
    
    def _create_filter(self, filter_conditions: Optional[Dict[str, Any]], collection_name: str, limit: int) -> Optional[Filter]:
        """
        Create a Qdrant filter from a dictionary of filter conditions.
        
        Args:
            filter_conditions: Dictionary mapping field names to values
            collection_name: Name of collection (for logging)
            limit: Search limit (for logging)
            
        Returns:
            Optional[Filter]: The created filter or None if no conditions
        """
        if not filter_conditions:
            logger.info(f"Searching in collection '{collection_name}' with limit {limit}")
            return None
            
        field_conditions = [
            FieldCondition(
                key=field,
                match=MatchValue(value=value)
            )
            for field, value in filter_conditions.items()
        ]
            
        logger.info(f"Searching in collection '{collection_name}' with filters: {filter_conditions} and limit {limit}")
        return Filter(must=field_conditions)
    
    def search(self, 
              collection_name: str, 
              query_vector: np.ndarray, 
              limit: int = 5,
              score_threshold: Optional[float] = None,
              filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar vectors in the collection, with optional filtering.
        
        Args:
            collection_name (str): The name of the collection to search in.
            query_vector (np.ndarray): The query vector to search for.
            limit (int): Maximum number of results to return.
            score_threshold (Optional[float]): Optional similarity threshold (0.0 to 1.0).
            filter_conditions (Optional[Dict[str, Any]]): Optional dictionary of field conditions for filtering.
            
        Returns:
            List[Dict]: List of search hits with payloads and scores.
        """
        # Create filter if conditions provided
        search_filter = self._create_filter(filter_conditions, collection_name, limit)
        
        # Set up search parameters
        search_params = {
            "collection_name": collection_name,
            "query": query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
            
        if search_filter is not None:
            search_params["query_filter"] = search_filter
            
        # Execute the search
        results = self.client.query_points(**search_params)
        return self._postprocess(results.points)
    
    def _postprocess(self, results: List[Dict], is_batch: bool = False) -> List[Dict]:
        """
        Postprocess the search results.
        """
        if not is_batch:
            return [result.__dict__ for result in results]
        else:
            return [[res.__dict__ for res in result.points] for result in results]

    def batch_search(self, 
                    collection_name: str,
                    query_vectors: List[np.ndarray],
                    score_threshold: Optional[float] = None,
                    filter_conditions: Optional[Dict[str, Any]] = None,
                    limit: int = 5) -> List[List[Dict]]:
        """
        Perform batch search for multiple query vectors.
        
        Args:
            collection_name (str): The name of the collection to search in.
            query_vectors (List[np.ndarray]): List of query vectors to search for.
            score_threshold (Optional[float]): Optional similarity threshold (0.0 to 1.0).
            filter_conditions (Optional[Dict[str, Any]]): Optional dictionary of field conditions for filtering.
            limit (int): Maximum number of results to return per query.
            
        Returns:
            List[List[Dict]]: List of search results for each query vector.
        """
        # Create filter if conditions provided
        search_filter = self._create_filter(filter_conditions, collection_name, limit)

        # Create search queries for each vector
        search_queries = [
            QueryRequest(
                query=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                score_threshold=score_threshold,
                filter=search_filter,
                limit=limit
            )
            for query_vector in query_vectors
        ]

        # Execute batch search
        results = self.client.query_batch_points(collection_name=collection_name, 
                                                requests=search_queries)
        return self._postprocess(results, is_batch=True)

# Example usage
if __name__ == "__main__":
    # Example data
    documents = [
        "Login and Account Issues",
        "Course Package and Activation Issues",
        "Installation and Device Compatibility Issues",
        "Payment and Subscription Issues",
        "Technical Support and Troubleshooting",
        "Account and Profile Management",
        "Content Access and Streaming Problems",
        "Device Compatibility and Installation Issues",
    ]
    payloads = [
        {"id": 1, "text": "Login and Account Issues"},
        {"id": 2, "text": "Course Package and Activation Issues"},
        {"id": 3, "text": "Installation and Device Compatibility Issues"},
        {"id": 4, "text": "Payment and Subscription Issues"},
        {"id": 5, "text": "Technical Support and Troubleshooting"},
        {"id": 6, "text": "Account and Profile Management"},
        {"id": 7, "text": "Content Access and Streaming Problems"},
        {"id": 8, "text": "Device Compatibility and Installation Issues"},
    ]

    from embeddings import EmbeddingFactory 
    model = EmbeddingFactory.create_provider(provider_type="openai", 
                                             model_name="text-embedding-3-small", 
                                             api_key=os.getenv("OPENAI_API_KEY"))

    # Generate embeddings
    # vectors = model.embed_documents(documents)
    
    # Initialize the vector store
    vector_store = VectorStore()
    
    # Create a collection
    # vector_store.create_collection(collection_name="test_collection", vector_size=model.dimension)
    
    # Insert vectors
    # vector_store.insert_vectors(
    #     collection_name="test_collection", 
    #     vectors=np.array(vectors), 
    #     payloads=payloads
    # )

    # Perform a search
    query = "I haven't accessed my app. I have blocked in login page"
    query_vector = model.embed_documents([query])[0]
    results = vector_store.search(
        collection_name="test_collection", 
        query_vector=query_vector, 
        limit=3
    )
    
    print("Search results:", results)
    
    # Filtered search example
    filtered_results = vector_store.search(
        collection_name="test_collection",
        query_vector=query_vector,
        filter_conditions={"id": 3},
        limit=1
    )
    
    print("\nFiltered search results:", filtered_results)

    # Batch search example
    query_vectors = [
        model.embed_documents([query])[0],
        model.embed_documents(["I have a problem with my account"])[0]
    ]
    batch_results = vector_store.batch_search(
        collection_name="test_collection",
        query_vectors=query_vectors,
        limit=2
    )

    print("\nBatch search results:", batch_results)
