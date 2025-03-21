import os
import uuid
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from .graph_extractor import GraphExtractor, KnowledgeTriplet, create_graph_extractor
from .neo4j_connection import Neo4jConnection
from .embeddings import create_embedding

logger = logging.getLogger(__name__)

class GraphRAG:
    """
    Graph RAG implementation using Neo4j for knowledge graph storage and retrieval
    
    This class implements the knowledge graph pattern for Retrieval Augmented Generation
    with optimized performance for both indexing and querying.
    """
    
    def __init__(
        self,
        # Neo4j configuration
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
        
        # Knowledge extraction
        graph_extractor_type: str = "openai",
        graph_extractor_model: str = "gpt-4o",
        graph_extractor_temperature: float = 0.1,
        
        # Embedding service
        embedding_provider_type: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = None,
        enable_caching: bool = True,
        cache_dir: str = ".embedding_cache",
        
        # Processing configuration
        max_workers: int = 4,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 100,
        
        # Query configuration
        semantic_search_limit: int = 10,
        graph_search_limit: int = 10,
        hybrid_search: bool = True,
        similarity_threshold: float = 0.7,
        
        # For backward compatibility
        neo4j_config: Dict[str, Any] = None,
        extractor_config: Dict[str, Any] = None,
        embedding_config: Dict[str, Any] = None
    ):
        """
        Initialize GraphRAG
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            graph_extractor_type: Type of extractor ("openai" or "gemini")
            graph_extractor_model: Model name for the extractor
            graph_extractor_temperature: Temperature for extractor
            embedding_provider_type: Type of embedding provider ("openai", "huggingface", or "google")
            embedding_model: Name of the embedding model
            embedding_dimension: Dimension of embeddings (optional - inferred from model if not provided)
            enable_caching: Whether to enable caching for embeddings
            cache_dir: Directory to store cached embeddings
            max_workers: Maximum number of concurrent workers for processing
            default_chunk_size: Default size of text chunks for processing
            default_chunk_overlap: Default overlap between text chunks
            semantic_search_limit: Maximum number of results from semantic search
            graph_search_limit: Maximum number of results from graph search
            hybrid_search: Whether to use hybrid search (semantic + graph)
            similarity_threshold: Threshold for semantic similarity
            neo4j_config: Legacy Neo4j configuration (for backward compatibility)
            extractor_config: Legacy extractor configuration (for backward compatibility)
            embedding_config: Legacy embedding configuration (for backward compatibility)
        """
        # Handle legacy configuration format
        if neo4j_config:
            neo4j_uri = neo4j_config.get("uri", neo4j_uri)
            neo4j_username = neo4j_config.get("username", neo4j_username)
            neo4j_password = neo4j_config.get("password", neo4j_password)
            neo4j_database = neo4j_config.get("database", neo4j_database)
            
        if extractor_config:
            graph_extractor_type = extractor_config.get("type", graph_extractor_type)
            graph_extractor_model = extractor_config.get("model", graph_extractor_model)
            graph_extractor_temperature = extractor_config.get("temperature", graph_extractor_temperature)
            
        if embedding_config:
            embedding_provider_type = embedding_config.get("type", embedding_provider_type)
            embedding_model = embedding_config.get("model", embedding_model)
            embedding_dimension = embedding_config.get("dimension", embedding_dimension)
        
        # Store configuration
        self.neo4j_config = {
            "uri": neo4j_uri,
            "username": neo4j_username,
            "password": neo4j_password,
            "database": neo4j_database
        }
        
        self.extractor_config = {
            "extractor_type": graph_extractor_type,
            "model": graph_extractor_model,
            "temperature": graph_extractor_temperature
        }
        
        self.embedding_config = {
            "provider_type": embedding_provider_type,
            "model_name": embedding_model,
            "dimensions": embedding_dimension,
            "cache": enable_caching,
            "cache_dir": cache_dir
        }
        
        # Processing settings
        self.max_workers = max_workers
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        
        # Query settings
        self.semantic_search_limit = semantic_search_limit
        self.graph_search_limit = graph_search_limit
        self.hybrid_search = hybrid_search
        self.similarity_threshold = similarity_threshold
        
        # Initialize components to None (they'll be created during initialization)
        self._neo4j = None
        self._extractor = None
        self._embedder = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize connections and components"""
        if not self._initialized:
            # Create Neo4j connection
            self._neo4j = Neo4jConnection(
                uri=self.neo4j_config["uri"],
                username=self.neo4j_config["username"],
                password=self.neo4j_config["password"],
                database=self.neo4j_config["database"],
                embedding_dim=self.embedding_config.get("dimensions")
            )
            
            # Connect to Neo4j and set up database
            await self._neo4j.connect()
            await self._neo4j.setup_database()
            
            # Create graph extractor
            self._extractor = create_graph_extractor(
                extractor_type=self.extractor_config["extractor_type"],
                model=self.extractor_config["model"],
                temperature=self.extractor_config["temperature"]
            )
            
            # Create embedding provider using direct vector_rag embeddings
            self._embedder = create_embedding(
                provider_type=self.embedding_config["provider_type"],
                model_name=self.embedding_config["model_name"],
                cache=self.embedding_config["cache"],
                cache_dir=self.embedding_config["cache_dir"]
            )
            
            # Update the Neo4j embedding dimension if not explicitly set
            if not self.embedding_config.get("dimensions"):
                self._neo4j.embedding_dim = self._embedder.dimension
            
            self._initialized = True
            logger.info("GraphRAG initialized successfully")
    
    async def close(self):
        """Close all connections"""
        if self._neo4j:
            await self._neo4j.close()
        
        if self._extractor and hasattr(self._extractor, 'close'):
            await self._extractor.close()
        
        self._initialized = False
        logger.info("GraphRAG connections closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def process_document(
        self,
        text: str,
        document_id: str = None,
        document_metadata: Dict[str, Any] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        extract_triplets: bool = True,
        add_embeddings: bool = True,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Process a document to extract knowledge and add to the graph
        
        Args:
            text: The document text
            document_id: Unique identifier for the document (generated if not provided)
            document_metadata: Additional metadata for the document
            chunk_size: Size of text chunks (default to class default)
            chunk_overlap: Overlap between text chunks (default to class default)
            extract_triplets: Whether to extract triplets from the text
            add_embeddings: Whether to add embeddings for entities
            batch_size: Batch size for database operations
            
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate document ID if not provided
        document_id = document_id or f"doc_{uuid.uuid4()}"
        document_metadata = document_metadata or {}
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        start_time = time.time()
        logger.info(f"Processing document {document_id} ({len(text)} characters)")
        
        # Split text into chunks
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        triplets = []
        if extract_triplets:
            # Extract triplets concurrently from chunks
            triplet_tasks = []
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Create task for triplet extraction
                task = asyncio.create_task(
                    self._extractor.extract_triplets(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page_number=i
                    )
                )
                triplet_tasks.append(task)
            
            # Gather all triplet extraction results
            chunk_triplets = await asyncio.gather(*triplet_tasks)
            
            # Flatten the list of lists into a single list of triplets
            triplets = [triplet for sublist in chunk_triplets for triplet in sublist]
            
            # Add triplets to Neo4j
            if triplets:
                await self._neo4j.add_triplets(triplets, batch_size=batch_size)
                logger.info(f"Added {len(triplets)} triplets to the graph")
        
        entity_embeddings = {}
        if add_embeddings and triplets:
            # Collect all unique entities
            entities = set()
            for triplet in triplets:
                entities.add(triplet.subject)
                entities.add(triplet.object)
            
            # Get embeddings for entities using direct vector_rag embeddings
            entity_embeddings = {}
            for entity in entities:
                entity_embeddings[entity] = self._embedder.embed_query(entity)
            
            # Add embeddings to Neo4j
            await self._neo4j.add_embeddings(entity_embeddings, batch_size=batch_size)
            logger.info(f"Added embeddings for {len(entity_embeddings)} entities")
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f}s")
        
        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "triplets": len(triplets),
            "entities": len(entity_embeddings) if entity_embeddings else 0,
            "processing_time": processing_time
        }
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        concurrency: int = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently
        
        Args:
            documents: List of document dictionaries with 'text' and optional fields
            concurrency: Maximum number of documents to process concurrently
            **kwargs: Additional arguments for process_document
            
        Returns:
            List of processing results
        """
        if not self._initialized:
            await self.initialize()
        
        concurrency = concurrency or min(self.max_workers, len(documents))
        logger.info(f"Processing {len(documents)} documents with concurrency {concurrency}")
        
        # Process documents in batches for better memory management
        results = []
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(doc):
            async with semaphore:
                text = doc.pop("text")
                doc_id = doc.pop("document_id", None)
                metadata = doc.pop("metadata", {})
                
                # Merge remaining doc fields with kwargs
                doc_kwargs = {**kwargs, **doc}
                
                return await self.process_document(
                    text=text,
                    document_id=doc_id,
                    document_metadata=metadata,
                    **doc_kwargs
                )
        
        # Start processing tasks
        tasks = [process_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def query(
        self,
        query_text: str,
        use_semantic_search: bool = True,
        use_graph_search: bool = True,
        combine_results: bool = True,
        semantic_search_limit: int = None,
        graph_search_limit: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph
        
        Args:
            query_text: The query text
            use_semantic_search: Whether to use semantic search
            use_graph_search: Whether to use graph-based search
            combine_results: Whether to combine and rank results
            semantic_search_limit: Maximum number of semantic search results
            graph_search_limit: Maximum number of graph search results
            similarity_threshold: Threshold for semantic similarity
            
        Returns:
            List of matching triplets
        """
        if not self._initialized:
            await self.initialize()
        
        # Use class defaults if not specified
        semantic_search_limit = semantic_search_limit or self.semantic_search_limit
        graph_search_limit = graph_search_limit or self.graph_search_limit
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # Use only specified search methods
        if not use_semantic_search and not use_graph_search:
            logger.warning("Neither semantic nor graph search enabled. Enabling graph search.")
            use_graph_search = True
        
        semantic_results = []
        graph_results = []
        
        # Run searches in parallel
        search_tasks = []
        
        if use_semantic_search:
            semantic_task = asyncio.create_task(
                self._perform_semantic_search(
                    query_text,
                    num_results=semantic_search_limit,
                    similarity_threshold=similarity_threshold
                )
            )
            search_tasks.append(semantic_task)
        
        if use_graph_search:
            graph_task = asyncio.create_task(
                self._perform_structured_search(
                    query_text,
                    num_results=graph_search_limit
                )
            )
            search_tasks.append(graph_task)
        
        # Wait for all search tasks to complete
        search_results = await asyncio.gather(*search_tasks)
        
        # Assign results based on which searches were enabled
        if use_semantic_search and use_graph_search:
            semantic_results = search_results[0]
            graph_results = search_results[1]
        elif use_semantic_search:
            semantic_results = search_results[0]
        elif use_graph_search:
            graph_results = search_results[0]
        
        # Combine results if needed
        if combine_results and semantic_results and graph_results:
            combined_results = self._combine_search_results(
                semantic_results,
                graph_results,
                num_results=max(semantic_search_limit, graph_search_limit)
            )
            return combined_results
        elif semantic_results:
            return semantic_results
        elif graph_results:
            return graph_results
        else:
            return []
    
    async def _perform_semantic_search(
        self,
        query_text: str,
        num_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using query embedding
        
        Args:
            query_text: The query text
            num_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching triplets
        """
        # Generate embedding for the query
        query_embedding = self._embedder.embed_query(query_text)
        
        # Perform semantic search
        semantic_entities = await self._neo4j.semantic_search(
            query_embedding=query_embedding,
            limit=num_results,
            similarity_threshold=similarity_threshold
        )
        
        # If no entities found, return empty list
        if not semantic_entities:
            return []
        
        # Get top matching entities
        top_entities = [entity["entity"] for entity in semantic_entities]
        
        # Get triplets for these entities
        results = []
        for entity in top_entities:
            # Query as subject
            subject_triplets = await self._neo4j.query_knowledge_graph(
                subject=entity,
                limit=num_results,
                exact_match=True
            )
            
            # Query as object
            object_triplets = await self._neo4j.query_knowledge_graph(
                object=entity,
                limit=num_results,
                exact_match=True
            )
            
            # Add to results
            results.extend(subject_triplets)
            results.extend(object_triplets)
        
        # Deduplicate by triplet_id and limit results
        seen_ids = set()
        deduplicated_results = []
        
        for result in results:
            triplet_id = result.get("triplet_id")
            if triplet_id not in seen_ids:
                seen_ids.add(triplet_id)
                deduplicated_results.append(result)
                
                if len(deduplicated_results) >= num_results:
                    break
        
        return deduplicated_results
    
    async def _perform_structured_search(
        self,
        query_text: str,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform structured search using extracted entities
        
        Args:
            query_text: The query text
            num_results: Maximum number of results
            
        Returns:
            List of matching triplets
        """
        # Extract entities and potential relations from the query
        query_triplets = await self._extractor.extract_triplets(query_text)
        
        # If no entities found in query, try more general search
        if not query_triplets:
            # Try direct keyword search (non-exact match)
            words = [w for w in query_text.split() if len(w) > 3]
            results = []
            
            for word in words:
                # Search for word as subject or object
                subject_results = await self._neo4j.query_knowledge_graph(
                    subject=word,
                    limit=num_results // 2,
                    exact_match=False
                )
                
                object_results = await self._neo4j.query_knowledge_graph(
                    object=word,
                    limit=num_results // 2,
                    exact_match=False
                )
                
                results.extend(subject_results)
                results.extend(object_results)
                
                if len(results) >= num_results:
                    break
            
            return results[:num_results]
        
        # Use the extracted entities for search
        results = []
        
        # Search for triplets matching query entities
        for triplet in query_triplets:
            # Search by subject
            if triplet.subject:
                subject_results = await self._neo4j.query_knowledge_graph(
                    subject=triplet.subject,
                    predicate=triplet.predicate if triplet.predicate else None,
                    object=triplet.object if triplet.object else None,
                    limit=num_results,
                    exact_match=False
                )
                results.extend(subject_results)
            
            # Search by object
            elif triplet.object:
                object_results = await self._neo4j.query_knowledge_graph(
                    object=triplet.object,
                    predicate=triplet.predicate if triplet.predicate else None,
                    limit=num_results,
                    exact_match=False
                )
                results.extend(object_results)
        
        # Deduplicate and limit results
        seen_ids = set()
        deduplicated_results = []
        
        for result in results:
            triplet_id = result.get("triplet_id")
            if triplet_id not in seen_ids:
                seen_ids.add(triplet_id)
                deduplicated_results.append(result)
                
                if len(deduplicated_results) >= num_results:
                    break
        
        return deduplicated_results
    
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
        # Create a map of triplet_id to result for deduplication
        combined_map = {}
        
        # Add semantic results with higher priority
        for result in semantic_results:
            triplet_id = result.get("triplet_id")
            if triplet_id and triplet_id not in combined_map:
                # Add source info for debugging/analytics
                result["source"] = "semantic"
                combined_map[triplet_id] = result
        
        # Add structured results
        for result in structured_results:
            triplet_id = result.get("triplet_id")
            if triplet_id and triplet_id not in combined_map:
                # Add source info for debugging/analytics
                result["source"] = "structured"
                combined_map[triplet_id] = result
        
        # Convert back to list and sort by confidence
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Limit results
        return combined_results[:num_results]
    
    def _split_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Handle edge cases
        if not text:
            return []
            
        if len(text) <= chunk_size:
            return [text]
        
        # Split text into chunks with respect to paragraph boundaries if possible
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = min(start + chunk_size, len(text))
            
            # Try to end at paragraph boundary if possible
            if end < len(text):
                # Look for paragraph breaks
                paragraph_break = max(
                    text.rfind("\n\n", start, end),
                    text.rfind("\r\n\r\n", start, end)
                )
                
                # If found, use it as the end
                if paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Otherwise look for sentence boundary
                    sentence_breaks = [text.rfind(". ", start, end),
                                     text.rfind("? ", start, end),
                                     text.rfind("! ", start, end),
                                     text.rfind(".\n", start, end)]
                    sentence_break = max(sentence_breaks)
                    
                    if sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move to next chunk, respecting overlap
            start = max(start + 1, end - chunk_overlap)
        
        return chunks 