import os
import uuid
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from llm_services import LLMProvider
from .graph_extractor import GraphExtractor, KnowledgeTriplet
from .neo4j_connection import SimpleNeo4jConnection
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Knowledge Graph Builder for constructing and maintaining a knowledge graph
    
    This class handles the process of building a knowledge graph from documents,
    extracting triplets, and storing them in Neo4j database.
    """
    
    def __init__(
        self,
        graph_store: SimpleNeo4jConnection,
        graph_extractor: GraphExtractor,
        embedding_provider: EmbeddingProvider,
        max_workers: int = 4,
        batch_size: int = 10,
       
    ):
        self._neo4j = graph_store 
        self._extractor = graph_extractor
        self._embedder = embedding_provider
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    async def close(self):
        """Close all connections"""
        if self._neo4j:
            self._neo4j.close()
        
        if self._extractor and hasattr(self._extractor, 'close'):
            await self._extractor.close()
        
        logger.info("GraphBuilder connections closed")
    
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def process_document(
        self,
        text: str,
        document_id: str = None,
        document_metadata: Dict[str, Any] = None,
        extract_triplets: bool = True,
        add_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Process a document to extract knowledge and add to the graph
        
        Args:
            text: The document text
            document_id: Unique identifier for the document (generated if not provided)
            document_metadata: Additional metadata for the document
            extract_triplets: Whether to extract triplets from the text
            add_embeddings: Whether to add embeddings for entities
            
        Returns:
            Processing results
        """
        # Generate document ID if not provided
        document_id = document_id or f"doc_{uuid.uuid4()}"
        document_metadata = document_metadata or {}
        
        start_time = time.time()
        logger.info(f"Processing document {document_id} ({len(text)} characters)")
        
        triplets = []
        if extract_triplets:
            # Extract triplets from document text
            triplets = await self._extractor.extract_triplets(
                text=text,
                metadata={
                    "document_id": document_id,
                    **document_metadata
                }
            )
            
            # Add triplets to Neo4j
            if triplets:
                extraction_results = [{
                    "text": text,
                    "metadata": {
                        "document_id": document_id,
                        "title": document_metadata.get("title", document_id),
                        **document_metadata
                    },
                    "triplets": triplets
                }]
                self._neo4j.add_extraction_triplet_results(extraction_results)
                logger.info(f"Added {len(triplets)} triplets to the graph")
        
        entity_embeddings = {}
        if add_embeddings and triplets:
            # Get embeddings for entities using direct vector_rag embeddings
            entity_embeddings = await self._extract_entity_embeddings(triplets)
            
            # Add embeddings to Neo4j
            if entity_embeddings:
                self._neo4j.add_entity_embeddings(entity_embeddings)
                logger.info(f"Added embeddings for {len(entity_embeddings)} entities")
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f}s")
        
        return {
            "document_id": document_id,
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
    
    async def _extract_entity_embeddings(self, triplets: List[Any]) -> Dict[str, List[float]]:
        """
        Extract entity embeddings from triplets
        
        Args:
            triplets: List of KnowledgeTriplet objects
            
        Returns:
            Dictionary mapping entity names to embedding vectors
        """
        # Extract unique entities
        entities = set()
        
        for triplet in triplets:
            if isinstance(triplet, KnowledgeTriplet):
                entities.add(triplet.subject)
                entities.add(triplet.object)
            elif isinstance(triplet, str):
                # Handle string representation of triplets
                # Format: "(subject) --[predicate]--> (object)"
                import re
                pattern = r"\((.*?)\)\s*--\[(.*?)\]-->\s*\((.*?)\)"
                match = re.search(pattern, triplet)
                if match:
                    entities.add(match.group(1).strip())
                    entities.add(match.group(3).strip())
        
        # Generate embeddings for entities
        entity_list = list(entities)
        entity_embeddings = {}
        
        embeddings = self._embedder.embed_documents(entity_list)
        for entity, embedding in zip(entity_list, embeddings):
            entity_embeddings[entity] = embedding
        
        return entity_embeddings 